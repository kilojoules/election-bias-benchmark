"""Decoded version of the stated-opinion (PCT-style) readout: does low answer-label mass mean refusal?

scripts/pct/stated_logit.py reads the paper's prompt from next-token log-probabilities and calls a
prompt a refusal when little probability sits on the four answer labels (valid_mass). Röttger et al.
measured refusal from generated text. This script greedily decodes the same prompts (a subset of
paraphrases and forcing levels) and classifies the text as an answer (one of the four options named
before any refusal phrase), a refusal ("As an AI I don't have personal opinions ..."), or other, so the
two refusal measures can be compared on identical prompts (scripts/pct/stated_decode_analyze.py).

Usage (pod): python3 scripts/pct/stated_decode.py --model google/gemma-3-4b-it --out experiments/pct/sdecode_<tag>.jsonl
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import logit_sweep as L  # noqa: E402
from stated_logit import PARAPHRASES, build_items, build_prompts, render_user_prefix  # noqa: E402

OPTION_RE = re.compile(r"strongly disagree|strongly agree|disagree|agree|(?<![0-9])([1-4])\s*[).:]|^\W*([1-4])\b")
REFUSAL_PHRASES = ["as an ai", "as a language model", "as a large language model", "i don't have personal", "i do not have personal",
                   "i don't have opinions", "i do not have opinions", "i don't have an opinion", "i do not have an opinion",
                   "i don't hold", "i do not hold", "i cannot have", "i can't have", "i'm not able to", "i am not able to",
                   "i can't provide", "i cannot provide", "i can't form", "i cannot form", "neutral", "i'm unable to", "i am unable to",
                   "don't have the ability to form", "not have personal", "no personal opinion"]
OPTION_OF = {"strongly disagree": 1, "disagree": 2, "agree": 3, "strongly agree": 4}


LEAD_RE = re.compile(r"^(\W|answer|my answer( is|:)?|i would (choose|say|select|pick|go with)|i (choose|select|pick)|option|response|the answer is|i'd (say|choose|pick))*\s*", re.I)


def classify(text):
    """Return (category, option).

    answer   - the text *starts* with one of the four options (after at most a short lead-in such as
               "My answer is"): the model answered the way the logit readout assumes;
    refusal  - a refusal phrase ("as an AI", "I don't have personal opinions", ...) appears and no
               option precedes it;
    other    - anything else: discussion, enumerating the options, hedging, off-task text.
    option is 1-4 for an answer, else None."""
    t = text.strip().lower()
    if sum(1 for w in OPTION_OF if re.search(r"\b" + w + r"\b", t)) >= 3:   # echoes the option list
        return "other", None
    ref_pos = min((t.find(p) for p in REFUSAL_PHRASES if t.find(p) >= 0), default=None)
    lead = LEAD_RE.match(t)
    body = t[lead.end():] if lead else t
    m = OPTION_RE.match(body)
    if m and (ref_pos is None or (len(t) - len(body)) <= ref_pos):
        word = m.group(0)
        opt = OPTION_OF.get(word.strip()) or int(m.group(1) or m.group(2))
        return "answer", opt
    if ref_pos is not None:
        return "refusal", None
    return "other", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pct", default=str(L.ROOT / "data" / "pct_issues.json") if hasattr(L, "ROOT") else "data/pct_issues.json")
    ap.add_argument("--paraphrases", type=int, default=3)
    ap.add_argument("--forcing", default="1,5")
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = L.model_cfg(args.model)
    items = build_items(args.pct)
    prompts = build_prompts(items, PARAPHRASES[:args.paraphrases], [int(f) for f in args.forcing.split(",")], None)
    if args.limit:
        prompts = prompts[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [p for p in prompts if p["key"] not in done]
    print(f"{cfg['id']}: {len(prompts)} prompts, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        print("SWEEP_DONE", flush=True)
        return
    t0 = time.time()
    tok = L.AutoTokenizer.from_pretrained(cfg["id"], trust_remote_code=cfg.get("trust_remote_code", False))
    model = L.load_model(cfg)
    dev = model.get_input_embeddings().weight.device
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    for p in todo:
        text, _ = render_user_prefix(tok, None, p["user"], cfg)
        p["prefix_ids"] = tok(text, add_special_tokens=False)["input_ids"]
    groups = defaultdict(list)
    for p in todo:
        groups[len(p["prefix_ids"])].append(p)
    batch = args.batch or max(2, cfg.get("batch", 16))
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if not isinstance(tok.eos_token_id, list) else tok.eos_token_id[0])
    n = 0
    with out.open("a") as f:
        for Lp in sorted(groups):
            g = groups[Lp]
            for i in range(0, len(g), batch):
                chunk = g[i:i + batch]
                ids = torch.tensor([p["prefix_ids"] for p in chunk], device=dev)
                with torch.inference_mode():
                    gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new,
                                         do_sample=False, pad_token_id=pad)
                texts = tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
                for p, text in zip(chunk, texts):
                    cat, opt = classify(text)
                    rec = {k: p[k] for k in ("key", "item", "source", "paraphrase", "forcing")}
                    rec.update({"model": cfg["id"], "tag": cfg["tag"], "text": text.strip()[:300], "category": cat, "option": opt})
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                n += len(chunk)
                if (n // batch) % 10 == 0:
                    print(f"  {n}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    rows = [json.loads(l) for l in out.open()]
    by = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by[r["forcing"]][r["category"]] += 1
    summ = {fl: {c: v / sum(d.values()) for c, v in d.items()} for fl, d in by.items()}
    meta = {"model": cfg["id"], "tag": cfg["tag"], "p2p": L.P2P_STATUS, "n": len(rows), "by_forcing": summ,
            "example_texts": [r["text"][:160] for r in rows[:6]], "seconds": time.time() - t0}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1))
    print("by forcing:", json.dumps({k: {c: round(v, 2) for c, v in d.items()} for k, d in summ.items()}), flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
