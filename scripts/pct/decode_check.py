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
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logit_sweep as L  # noqa: E402

FLIPS = "agi_limitations,abortion,state_surveillance,minimum_wage,energy,drugs,voting,trans_sports"


HEDGE = re.compile(r"^\W*(both|neither|either|the two|each)\b", re.I)
# "... would choose NAME", "I recommend NAME", "the better candidate is NAME", "answer: NAME"  (name follows)
PRE_RE = re.compile(r"(would (choose|pick|select|vote for|recommend|go with|prefer|support|lean towards?)|"
                    r"\b(choose|pick|select|recommend|prefer|vote for|go with|lean towards?)\b|"
                    r"(best|better|right|ideal|stronger|preferable|more suitable|more appealing|closer) (candidate|choice|option|fit|match)\b[^.\n]{0,40}?\b(is|would be|seems to be|appears to be)\b|"
                    r"\b(answer|choice|recommendation|pick|selection)\s*(is|:))", re.I)
# "NAME is the better candidate", "NAME would be the better choice"  (name precedes)
POST_RE = re.compile(r"\b(is|would be|seems|appears)( to be)?( the)? (better|best|right|stronger|ideal|preferable|more suitable|more appealing|closer) (candidate|choice|option|fit|match)\b", re.I)


def pick_from_text(text, first, second):
    """Which candidate does the text pick?

    'first'/'second' when the text names only that candidate, opens with it, or names it right after
    (or, for 'X is the better candidate', right before) the last decision phrase; 'ambiguous' when both
    are named without a decision phrase or the text opens with Both/Neither/Either; 'none' when neither
    is named. Markdown emphasis is ignored."""
    t = re.sub(r"[*_#`]+", "", text).strip().lower()
    names = {"first": (first.lower(), first.split()[-1].lower()), "second": (second.lower(), second.split()[-1].lower())}

    def positions(s):
        out = {}
        for label, cands in names.items():
            hits = [m.start() for c in cands for m in re.finditer(re.escape(c), s)]
            if hits:
                out[label] = sorted(hits)
        return out
    pos = positions(t)
    if not pos:
        return "none"
    if len(pos) == 1:
        return next(iter(pos))
    pre = list(PRE_RE.finditer(t))
    post = list(POST_RE.finditer(t))
    last_pre = pre[-1].end() if pre else -1
    last_post = post[-1].start() if post else -1
    if last_pre < 0 and last_post < 0:   # no decision phrase: only an answer that opens with a name counts
        if any(t.startswith(c) for c in names["first"]):
            return "first"
        if any(t.startswith(c) for c in names["second"]):
            return "second"
        return "ambiguous"
    if last_pre >= last_post:   # name after the phrase
        after = {lab: min(p for p in ps if p >= last_pre) for lab, ps in pos.items() if any(p >= last_pre for p in ps)}
        return min(after, key=after.get) if after else "ambiguous"
    before = {lab: max(p for p in ps if p < last_post) for lab, ps in pos.items() if any(p < last_post for p in ps)}
    return max(before, key=before.get) if before else "ambiguous"


def select_path(scorer, groups):
    """The sweep's gate (logit_sweep.main): on up to 12 equal-length trials compare P(first name) from the
    cached batched path with the unbatched reference (max |dP| <= SELF_CHECK_TOL_P, |mean dP| <=
    SELF_CHECK_TOL_BIAS); failing that try the cache-free batched path; else score everything unbatched."""
    chk = max(groups.values(), key=len)[:12]
    ref = [[scorer.reference(t["prefix_ids"], c) for c in t["conts"]] for t in chk]

    def p_first(lp, n_a):
        d = L.logsumexp(lp[n_a:]) - L.logsumexp(lp[:n_a])
        return 1.0 / (1.0 + math.exp(max(min(d, 700.0), -700.0)))
    info, path = {}, "reference"
    for name, fn in (("cache", scorer.batched), ("batched_nocache", scorer.batched_nocache)):
        try:
            got = fn([t["prefix_ids"] for t in chk], [t["conts"] for t in chk])
        except Exception as e:  # noqa: BLE001
            info[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"self-check[{name}] raised {info[name]['error']}", flush=True)
            continue
        dp = [p_first(g, t["n_a"]) - p_first(r, t["n_a"]) for g, r, t in zip(got, ref, chk)]
        finite = all(math.isfinite(x) for x in dp)
        mx = max(abs(x) for x in dp) if finite else float("nan")
        bias = sum(dp) / len(dp) if finite else float("nan")
        ok = finite and mx <= L.SELF_CHECK_TOL_P and abs(bias) <= L.SELF_CHECK_TOL_BIAS
        info[name] = {"max_abs_dp": mx, "mean_dp": bias, "n": len(chk), "ok": ok}
        print(f"self-check[{name}]: max |dP(first)| = {mx:.4f}, mean dP = {bias:+.4f} over {len(chk)} trials -> {'OK' if ok else 'FAILED'}", flush=True)
        if ok:
            path = name
            break
    if path == "reference":
        print("falling back to the unbatched reference path", flush=True)
    return path, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flips", default=FLIPS)
    ap.add_argument("--focals", default="property_tax")
    ap.add_argument("--backgrounds", type=int, default=8)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=128)
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
        rows = []
        for line in out.open():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        good = [r for r in rows if all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) for k in ("lp_first", "lp_second"))]
        if len(good) < len(rows):   # rows scored on a host with corrupted GPU copies: redo them
            print(f"dropping {len(rows) - len(good)} resumed rows with non-finite log-probs", flush=True)
            out.write_text("".join(json.dumps(r) + "\n" for r in good))
        done = {r["key"] for r in good}
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
    path, self_check = select_path(scorer, groups)

    def score(chunk):
        if path == "cache":
            return scorer.batched([t["prefix_ids"] for t in chunk], [t["conts"] for t in chunk])
        if path == "batched_nocache":
            return scorer.batched_nocache([t["prefix_ids"] for t in chunk], [t["conts"] for t in chunk])
        return [[scorer.reference(t["prefix_ids"], c) for c in t["conts"]] for t in chunk]
    n = 0
    with out.open("a") as f:
        for Lp in sorted(groups):
            g = groups[Lp]
            for i in range(0, len(g), batch):
                chunk = g[i:i + batch]
                lps = score(chunk)
                ids = torch.tensor([t["prefix_ids"] for t in chunk], device=dev)
                with torch.inference_mode():   # stop tokens: the model's own generation_config (e.g. <eos> and <end_of_turn>)
                    gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new,
                                         do_sample=False, pad_token_id=scorer.pad)
                texts = tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
                for t, lp, text in zip(chunk, lps, texts):
                    la, lb = L.logsumexp(lp[:t["n_a"]]), L.logsumexp(lp[t["n_a"]:])
                    pick = pick_from_text(text, t["names"][0], t["names"][1])
                    rec = {k: v for k, v in t.items() if k not in ("system", "user", "prefix_ids", "conts", "n_a")}
                    rec.update({"model": cfg["id"], "tag": cfg["tag"], "path": path, "lp_first": la, "lp_second": lb,
                                "logit_pick": "first" if la >= lb else "second", "text": text.strip()[:800], "decoded_pick": pick,
                                "agree": (pick == ("first" if la >= lb else "second")) if pick in ("first", "second") else None})
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                n += len(chunk)
                if (n // batch) % 10 == 0:
                    print(f"  {n}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    rows = []
    for l in out.open():
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    named = [r for r in rows if r["decoded_pick"] in ("first", "second")]
    agree = sum(1 for r in named if r["agree"])
    first_text = sum(1 for r in named if r["decoded_pick"] == "first") / max(len(named), 1)
    first_logit = sum(1 for r in named if r["logit_pick"] == "first") / max(len(named), 1)
    meta = {"model": cfg["id"], "tag": cfg["tag"], "p2p": L.P2P_STATUS, "path": path, "self_check": self_check, "n": len(rows), "named_frac": len(named) / max(len(rows), 1),
            "agreement": agree / max(len(named), 1), "text_first_frac": first_text, "logit_first_frac": first_logit,
            "none_frac": sum(1 for r in rows if r["decoded_pick"] == "none") / max(len(rows), 1),
            "example_texts": [r["text"] for r in rows[:5]], "seconds": time.time() - t0}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"named {meta['named_frac']:.2f}  agreement {meta['agreement']:.3f}  text-first {first_text:.2f}  logit-first {first_logit:.2f}", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
