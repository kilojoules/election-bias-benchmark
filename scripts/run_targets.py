"""Generate responses from a list of target models. Runs on the pod.

Usage: python3 run_targets.py --models google/gemma-3-1b-it,google/gemma-3-4b-it --n 10 --out experiments/responses.jsonl
Writes one JSON line per (model, job). Resumes: skips (model, job_key) pairs already present in --out.
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_jobs  # noqa: E402


def job_key(j):
    if j["kind"] == "pairwise":
        return f"{j['prompt_id']}|{j['first_id']}|{j['second_id']}|{j['sample']}"
    return f"{j['prompt_id']}|{j['candidate_id']}|{j['sample']}"


def load_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    except Exception as e:  # multimodal gemma-3 checkpoints
        print(f"AutoModelForCausalLM failed ({type(e).__name__}); trying Gemma3ForConditionalGeneration", flush=True)
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.eval()
    return tok, model


def detect_merge_sys(tok):
    try:
        tok.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}], tokenize=False)
        return False
    except Exception:
        return True


def make_chat(tok, system, user, merge_sys):
    if merge_sys:
        msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
    else:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def free_model_cache(model_id):
    d = Path.home() / ".cache/huggingface/hub" / ("models--" + model_id.replace("/", "--"))
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        print(f"freed cache {d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--free_cache", action="store_true", help="delete each model's HF cache after use")
    ap.add_argument("--prompts", default=None, help="comma-separated prompt ids to run; default all")
    ap.add_argument("--system_extra", default=None, help="text appended to the system prompt (e.g. a hidden-preference instruction for a positive control)")
    ap.add_argument("--model_tag", default="", help="suffix added to the model id in output records, e.g. +hidden_pro_ai")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)
            done.add((r["model"], r["job_key"]))
    jobs = build_jobs(args.n, seed=args.seed)
    if args.prompts:
        keep = set(args.prompts.split(","))
        jobs = [j for j in jobs if j["prompt_id"] in keep]
    print(f"{len(jobs)} jobs per model; {len(done)} already done", flush=True)

    for model_id in args.models.split(","):
        todo = [j for j in jobs if (model_id + args.model_tag, job_key(j)) not in done]
        if not todo:
            print(f"{model_id}: nothing to do", flush=True)
            continue
        print(f"=== {model_id}: {len(todo)} generations", flush=True)
        t0 = time.time()
        tok, model = load_model(model_id)
        merge_sys = detect_merge_sys(tok)
        print(f"loaded in {time.time()-t0:.0f}s; merge_sys={merge_sys}", flush=True)
        torch.manual_seed(args.seed)
        # Sort by prompt length so batches pad less.
        todo.sort(key=lambda j: len(j["user"]))
        with out.open("a") as f:
            for i in range(0, len(todo), args.batch):
                chunk = todo[i:i + args.batch]
                texts = [make_chat(tok, j["system"] + ("\n\n" + args.system_extra if args.system_extra else ""), j["user"], merge_sys) for j in chunk]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                         temperature=args.temperature, top_p=0.95,
                                         pad_token_id=tok.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                decoded = tok.batch_decode(new, skip_special_tokens=True)
                for j, resp in zip(chunk, decoded):
                    rec = {k: v for k, v in j.items() if k not in ("system", "user")}
                    rec.update({"model": model_id + args.model_tag, "job_key": job_key(j), "response": resp.strip(),
                                "n_words": len(resp.split()), "merge_sys": merge_sys,
                                "system_extra": args.system_extra})
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                if (i // args.batch) % 10 == 0:
                    print(f"  {i+len(chunk)}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
        print(f"{model_id} done in {time.time()-t0:.0f}s", flush=True)
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        if args.free_cache:
            free_model_cache(model_id)
    print("TARGETS_DONE", flush=True)


if __name__ == "__main__":
    main()
