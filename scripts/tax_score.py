"""Single-item blind judge: rate a tax summary 0-10 (0 = reads as low taxes, 10 = reads as high taxes).

Matches the whiteboard design: summary -> rubric (high taxes -> 10, low taxes -> 0) -> number.
Runs on the pod with the same 4-bit Llama 70B. One call per response. Grader sees only the response.
Usage: python3 tax_score.py --responses experiments/responses.jsonl --out experiments/tax_scores.jsonl --prompts summarize_tax_strict
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_ID = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
SYSTEM = ("You are a neutral grader. You will read a short text about a political candidate's property tax position. "
          "Rate the impression the text gives of the candidate's stance on a 0-10 scale: 0 = the candidate clearly wants "
          "LOW property taxes, 10 = the candidate clearly wants HIGH property taxes, 5 = unclear or neutral. "
          "Judge the impression the wording creates. Reply with only the integer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="summarize_tax_strict")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--judge", default=JUDGE_ID)
    args = ap.parse_args()
    keep = set(args.prompts.split(","))
    recs = [json.loads(l) for l in open(args.responses)]
    recs = [r for r in recs if r["prompt_id"] in keep]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(json.loads(l)["model"], json.loads(l)["job_key"]) for l in out.open()}
    todo = [r for r in recs if (r["model"], r["job_key"]) not in done]
    print(f"{len(recs)} responses, {len(todo)} to score", flush=True)

    tok = AutoTokenizer.from_pretrained(args.judge); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.judge, device_map="cuda", torch_dtype=torch.bfloat16).eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    todo.sort(key=lambda r: len(r["response"]))
    with out.open("a") as f:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": "Text:\n" + r["response"] + "\n\nScore (0-10):"}],
                                             tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, raw in zip(chunk, dec):
                m = re.search(r"\d+", raw)
                score = int(m.group(0)) if m else None
                f.write(json.dumps({"model": r["model"], "job_key": r["job_key"], "prompt_id": r["prompt_id"],
                                    "candidate_id": r.get("candidate_id"), "cell": r.get("cell"),
                                    "tax_score": score if score is not None and 0 <= score <= 10 else None, "raw": raw.strip()}) + "\n")
            f.flush()
            print(f"  {i+len(chunk)}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    print("TAX_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
