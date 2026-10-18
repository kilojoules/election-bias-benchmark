"""Does the teacher-forced name readout agree with what the model actually writes?

For a sample of conjoint trials the script (1) scores both names from logits exactly as
scripts/logit_sweep.py does and (2) greedily decodes the model's answer, then records which candidate
the text names first. Reports agreement between the decoded pick and the logit argmax, the fraction
of trials in which the text names a candidate at all (relevant for the models excluded by naming
mass), and the listing-position effect in text vs logits. This is the "first-token probabilities do
not match text answers" objection (Wang et al. 2024b, cited by Röttger et al.) checked directly.

Usage (pod): python3 scripts/pct/decode_check.py --model google/gemma-3-4b-it --out experiments/pct/decode_<tag>.jsonl
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
import logit_sweep as L  # noqa: E402

FLIPS = "agi_limitations,abortion,state_surveillance,minimum_wage,energy,drugs,voting,trans_sports"


def pick_from_text(text, first, second):
    """Which candidate does the text name first? Full name or surname, case-insensitive."""
    t = text.lower()
    pos = {}
    for label, name in (("first", first), ("second", second)):
        cands = [name.lower(), name.split()[-1].lower()]
        hits = [t.find(c) for c in cands if t.find(c) >= 0]
        if hits:
            pos[label] = min(hits)
    if not pos:
        return "none"
    if len(pos) == 2 and pos["first"] == pos["second"]:
        return "ambiguous"
    return min(pos, key=pos.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flips", default=FLIPS)
    ap.add_argument("--focals", default="property_tax")
    ap.add_argument("--backgrounds", type=int, default=8)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = L.model_cfg(args.model)
    stances, cands, prompts = L.load_data()
    trials = L.build_sweep_trials(args.focals.split(","), args.flips.split(","), args.backgrounds, args.draws,
                                  L.ISSUES["n_fillers"], stances["name_pool"]["names"], prompts["system"], prompts["profile_template"], 0)
    if args.limit:
        trials = trials[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                done.add(json.loads(line)["key"])
            except json.JSONDecodeError:
                pass
    todo = [t for t in trials if t["key"] not in done]
    print(f"{cfg['id']}: {len(trials)} trials, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        print("SWEEP_DONE", flush=True)
        return

    t0 = time.time()
    tok = L.AutoTokenizer.from_pretrained(cfg["id"], trust_remote_code=cfg.get("trust_remote_code", False))
    model = L.load_model(cfg)
    dev = model.get_input_embeddings().weight.device
    scorer = L.Scorer(model, tok, dev)
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    for t in todo:
        text, _ = L.render_prefix(tok, t["system"], t["user"], cfg)
        t["prefix_ids"] = tok(text, add_special_tokens=False)["input_ids"]
        t["conts"] = L.name_variants(tok, t["names"][0]) + L.name_variants(tok, t["names"][1])
        t["n_a"] = len(L.name_variants(tok, t["names"][0]))
    groups = defaultdict(list)
    for t in todo:
        groups[len(t["prefix_ids"])].append(t)
    batch = args.batch or max(2, cfg.get("batch", 16) // 2)
    eos = tok.eos_token_id if not isinstance(tok.eos_token_id, list) else tok.eos_token_id[0]
    n = 0
    with out.open("a") as f:
        for Lp in sorted(groups):
            g = groups[Lp]
            for i in range(0, len(g), batch):
                chunk = g[i:i + batch]
                lps = scorer.batched([t["prefix_ids"] for t in chunk], [t["conts"] for t in chunk])
                ids = torch.tensor([t["prefix_ids"] for t in chunk], device=dev)
                with torch.inference_mode():
                    gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new,
                                         do_sample=False, pad_token_id=scorer.pad, eos_token_id=eos)
                texts = tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
                for t, lp, text in zip(chunk, lps, texts):
                    la, lb = L.logsumexp(lp[:t["n_a"]]), L.logsumexp(lp[t["n_a"]:])
                    pick = pick_from_text(text, t["names"][0], t["names"][1])
                    rec = {k: v for k, v in t.items() if k not in ("system", "user", "prefix_ids", "conts", "n_a")}
                    rec.update({"model": cfg["id"], "tag": cfg["tag"], "lp_first": la, "lp_second": lb,
                                "logit_pick": "first" if la >= lb else "second", "text": text.strip()[:200], "decoded_pick": pick,
                                "agree": (pick == ("first" if la >= lb else "second")) if pick in ("first", "second") else None})
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                n += len(chunk)
                if (n // batch) % 10 == 0:
                    print(f"  {n}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    rows = [json.loads(l) for l in out.open()]
    named = [r for r in rows if r["decoded_pick"] in ("first", "second")]
    agree = sum(1 for r in named if r["agree"])
    first_text = sum(1 for r in named if r["decoded_pick"] == "first") / max(len(named), 1)
    first_logit = sum(1 for r in rows if r["logit_pick"] == "first") / max(len(rows), 1)
    meta = {"model": cfg["id"], "tag": cfg["tag"], "n": len(rows), "named_frac": len(named) / max(len(rows), 1),
            "agreement": agree / max(len(named), 1), "text_first_frac": first_text, "logit_first_frac": first_logit,
            "none_frac": sum(1 for r in rows if r["decoded_pick"] == "none") / max(len(rows), 1),
            "example_texts": [r["text"] for r in rows[:5]], "seconds": time.time() - t0}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"named {meta['named_frac']:.2f}  agreement {meta['agreement']:.3f}  text-first {first_text:.2f}  logit-first {first_logit:.2f}", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
