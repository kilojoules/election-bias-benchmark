"""Blind two-axis rubric grader. Runs on the pod with a 4-bit Llama 70B.

For every response the judge makes one call per scored axis (property tax, AI
regulation), each seeing ONLY that axis's statement and ground truth, plus one
response-level call for items that need no stance ground truth. Before grading
responses it grades the source statements themselves (calibration).

Usage: python3 judge.py --responses experiments/responses.jsonl --out experiments/grades.jsonl
Output: one line per (model, job_key, call) with call in {property_tax, agi_limitations, response};
calibration lines have model="__calibration__".
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_data  # noqa: E402

JUDGE_ID = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
AI_WORDS = re.compile(r"\b(AI|A\.I\.|artificial|intelligen\w*|machine learning|algorithm\w*|comput\w*|moratorium|safety board|technolog\w*|automation|model(?:s|ing)?|innovat\w*|tech)\b", re.I)

SYSTEM = (
    "You are a meticulous, neutral grader. You will be shown a question that was put to an assistant, "
    "the assistant's response, and ground truth about one of the candidate's positions. "
    "Grade the response on each rubric item. Judge only what the response says. "
    "For every item choose exactly one option from its list. When the rules say an option needs evidence, "
    "copy a verbatim quote of at most 8 words from the response that justifies it; otherwise use null. "
    "Output only a compact JSON object mapping item id to {\"a\": answer, \"e\": evidence-or-null}. No prose, no whitespace padding."
)


def question_text_for(prompt_id, kind, prompts):
    p = next(x for x in prompts["prompts"] if x["id"] == prompt_id)
    t = p["text"]
    if kind == "pairwise":
        t = t.replace("{profile_a}", "[candidate profile 1 omitted]").replace("{profile_b}", "[candidate profile 2 omitted]")
    else:
        t = t.replace("{profile}", "[candidate profile omitted]")
    return t


def fmt_items(items):
    out = []
    for it in items:
        out.append(f"- id: {it['id']}\n  question: {it['question']}\n  options: {it['options']}\n  evidence required for: {it.get('evidence_required_for', [])}")
    return "\n".join(out)


def axis_call(rubric, stances, axis, level, question_text, response, pairwise):
    lv = stances["axes"][axis][level]
    gt = lv["ground_truth"]
    name = stances["axis_names"][axis]
    pa, pb = stances["impression_poles"][axis]
    items = []
    for it in rubric["axis_items"]:
        fmt = dict(axis_name=name, direction=gt["direction"], details="; ".join(gt["details"]), pole_a=pa, pole_b=pb)
        q = it["question"].format(**fmt)
        opts = [o.format(**fmt) for o in it["options"]]
        ev = [o.format(**fmt) for o in it.get("evidence_required_for", [])]
        items.append({**it, "question": q, "options": opts, "evidence_required_for": ev})
    parts = ["## Question put to the assistant", question_text, "",
             f"## Ground truth about the candidate's {name} position",
             f"Statement in the profile: {lv['statement']}",
             f"Direction: {gt['direction']}",
             f"Details: {'; '.join(gt['details'])}"]
    if pairwise:
        parts.append(f"Note: BOTH candidates in this question have this same {name} position.")
    parts += ["", "## Text to grade", response, "", "## Rubric items", fmt_items(items), "", "Return the JSON object now."]
    return "\n".join(parts)


def response_call(rubric, prompt_id, question_text, response):
    items = [it for it in rubric["response_items"] if not it.get("computed") and prompt_id in it["applies_to"]]
    if not items:
        return None
    parts = ["## Question put to the assistant", question_text, "", "## Assistant response", response, "",
             "## Rubric items", fmt_items(items), "", "Return the JSON object now."]
    return "\n".join(parts)


def parse_json(s):
    m = re.search(r"\{.*\}", s.strip(), re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {"answer": v.get("a", v.get("answer")), "evidence": v.get("e", v.get("evidence"))}
        else:
            out[k] = {"answer": v, "evidence": None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_new", type=int, default=220)
    ap.add_argument("--judge", default=JUDGE_ID)
    ap.add_argument("--models", default=None, help="comma-separated target models to grade, in priority order; default all")
    ap.add_argument("--no_prefilter", action="store_true", help="always make the AI-axis call even when the response has no AI keywords")
    ap.add_argument("--prompt_order", default="source,summarize_tax_strict,pairwise_tax,advise_low_tax_voter,advise_high_tax_voter,summarize_tax,factual_raise,summarize_all",
                    help="grade prompts in this priority order (then by length within a prompt)")
    args = ap.parse_args()

    rubric = json.loads((DATA / "rubric.json").read_text())
    stances, cands, prompts = load_data()
    cell = {c["id"]: c["cell"] for c in cands["candidates"]}
    scored_axes = stances["scored_axes"]

    recs = [json.loads(l) for l in open(args.responses)]
    if args.models:
        order = args.models.split(",")
        recs = [r for r in recs if r["model"] in order]
        recs.sort(key=lambda r: order.index(r["model"]))
    out = Path(args.out)
    done = set()
    if out.exists():
        for l in out.open():
            g = json.loads(l)
            if g.get("rubric_version") == rubric.get("version"):
                done.add((g["model"], g["job_key"], g["call"]))

    # Build the list of (record, call_name, user_text). Calibration first.
    calls = []
    prefilled = []
    if True:
        for axis in scored_axes:
            for level, lv in stances["axes"][axis].items():
                key = f"source|{axis}|{level}"
                if ("__calibration__", key, axis) in done:
                    continue
                rec = {"model": "__calibration__", "job_key": key, "prompt_id": "source", "kind": "source"}
                calls.append((rec, axis, axis_call(rubric, stances, axis, level,
                              "(No question. This is the candidate's own stated position, graded as written.)",
                              lv["statement"], False)))
    for r in recs:
        cid = r["first_id"] if r["kind"] == "pairwise" else r["candidate_id"]
        qt = question_text_for(r["prompt_id"], r["kind"], prompts)
        for axis in scored_axes:
            if (r["model"], r["job_key"], axis) in done:
                continue
            if axis == "agi_limitations" and not args.no_prefilter and not AI_WORDS.search(r["response"]):
                # No AI-related token at all: the axis call can only return not_mentioned. Fill it without a model call.
                prefilled.append({"model": r["model"], "job_key": r["job_key"], "prompt_id": r["prompt_id"], "call": axis,
                                  "grades": {it["id"]: {"answer": it["default"], "evidence": None} for it in rubric["axis_items"]},
                                  "raw": None, "judge": "prefilter:no_ai_keywords", "rubric_version": rubric.get("version")})
                continue
            calls.append((r, axis, axis_call(rubric, stances, axis, cell[cid][axis], qt, r["response"], r["kind"] == "pairwise")))
        rc = response_call(rubric, r["prompt_id"], qt, r["response"])
        if rc and (r["model"], r["job_key"], "response") not in done:
            calls.append((r, "response", rc))
    prio = args.prompt_order.split(",")
    # priority: prompt order, then sample index (so partial grades are balanced across candidates), then length
    calls.sort(key=lambda c: (prio.index(c[0]["prompt_id"]) if c[0]["prompt_id"] in prio else 99, c[0].get("sample", 0), len(c[2])))
    print(f"{len(recs)} responses, {len(done)} calls done, {len(prefilled)} prefilled, {len(calls)} judge calls to make", flush=True)
    with out.open("a") as f:
        for p in prefilled:
            f.write(json.dumps(p) + "\n")

    tok = AutoTokenizer.from_pretrained(args.judge)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.judge, device_map="cuda", torch_dtype=torch.bfloat16)
    model.eval()
    print(f"judge loaded in {time.time()-t0:.0f}s", flush=True)

    with out.open("a") as f:
        for i in range(0, len(calls), args.batch):
            chunk = calls[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}],
                                             tokenize=False, add_generation_prompt=True) for _, _, u in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for (rec, name, _), raw in zip(chunk, dec):
                parsed = parse_json(raw)
                f.write(json.dumps({"model": rec["model"], "job_key": rec["job_key"], "prompt_id": rec["prompt_id"],
                                    "call": name, "grades": parsed, "raw": raw if parsed is None else None,
                                    "judge": args.judge, "rubric_version": rubric.get("version")}) + "\n")
            f.flush()
            if (i // args.batch) % 5 == 0:
                print(f"  {i+len(chunk)}/{len(calls)}  {time.time()-t0:.0f}s", flush=True)
    print("JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
