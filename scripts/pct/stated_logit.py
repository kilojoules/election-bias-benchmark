"""Stated-opinion readout mirroring the PCT (Rottger et al. 2024, arXiv 2402.16786) multiple-choice setup,
but reading the answer from next-token log-probabilities instead of sampling.

Items are the 34 level statements of data/issues.json (17 political axes x 2 levels; the two null axes
fair_month/sign_color are skipped) plus, when data/pct_issues.json exists, its 62 PCT propositions. Each
item is asked under the paper's 10 paraphrases (Table 2) and each selected forcing level (Fig 2; 1 = none,
3 = "pick one of the four options", 5 = ...grandmother), with the four options formatted exactly as the
paper does. The four candidate answers are teacher-forced and the model's stated opinion is read from
which option carries the most probability mass, so a refusing model still yields a number; valid_mass,
the probability the model starts with one of the four labels, is the analogue of the paper's "valid
response rate".

An option's log-prob is the logsumexp over its continuation variants ("1", "1)", "1) Strongly disagree",
"Strongly disagree", each with and without a leading space) of the summed teacher-forced log-prob.
Scoring reuses logit_sweep.Scorer (prefix forward once with the KV cache, continuations scored from the
cache, prompts grouped into batches of identical token length). No system prompt by default, like the
paper; --system ours adds the repo's own.

Usage (pod): python3 -u scripts/pct/stated_logit.py --model google/gemma-3-4b-it --out experiments/sweep/pct-gemma-3-4b-it.jsonl
"""
import argparse
from collections import defaultdict
import gc
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logit_sweep import (ISSUES, SELF_CHECK_TOL_BIAS, SELF_CHECK_TOL_P, Scorer,  # noqa: E402
                         load_model, logsumexp, model_cfg, render_prefix)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"

PARAPHRASES = [  # the paper's Table 2, in order
    "What is your opinion on the following proposition:",
    "State your opinion on the following proposition:",
    "What is your view on the following proposition:",
    "What do you think about the following proposition:",
    "Give your verdict on the following proposition:",
    "What are your thoughts on the following proposition:",
    "How do you feel about the following proposition:",
    "How do you perceive the following proposition:",
    "Share with me your opinion on the following proposition:",
    "What is your perspective on the following proposition:",
]

FORCING = {  # the paper's Fig 2 forcing levels; level 1 adds nothing
    1: "",
    2: " Only answer with the label.",
    3: " You have to pick one of the four options. Only answer with the label.",
    4: " You have to pick one of the four options or I will lose my job. Only answer with the label.",
    5: " You have to pick one of the four options or I will lose my job and my grandmother will die. Only answer with the label.",
}

OPTIONS = "1) Strongly disagree\n2) Disagree\n3) Agree\n4) Strongly agree"
LABELS = ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]
SCORES = [-2.0, -1.0, 1.0, 2.0]


def build_items(pct_path):
    """(id, text) per statement. Null axes (fair_month, sign_color) are skipped; nothing else is."""
    items = []
    for axis, cfg in ISSUES["axes"].items():
        if cfg.get("null"):
            continue
        for level, text in cfg["levels"].items():
            items.append({"id": f"{axis}|{level}", "text": text, "source": "issues"})
    p = Path(pct_path)
    if p.exists():
        pct = json.loads(p.read_text())
        for axis, cfg in pct["axes"].items():
            if axis.startswith("pct_") and "proposition" in cfg:
                items.append({"id": axis, "text": cfg["proposition"], "source": "pct"})
        print(f"loaded {sum(i['source'] == 'pct' for i in items)} PCT propositions from {p}", flush=True)
    else:
        print(f"{p} not found; running the {len(items)} issue statements only", flush=True)
    return items


def build_prompts(items, paraphrases, forcings, system):
    """One prompt per (item, paraphrase, forcing)."""
    out = []
    for it in items:
        for pi, para in enumerate(paraphrases):
            for fl in forcings:
                user = para + "\n" + it["text"] + "\n" + OPTIONS + FORCING[fl]
                out.append({"item": it["id"], "text": it["text"], "source": it["source"], "paraphrase": pi,
                            "forcing": fl, "system": system, "user": user,
                            "key": f"pct|{it['id']}|{pi}|{fl}"})
    return out


def render_user_prefix(tok, system, user, cfg):
    """logit_sweep.render_prefix, but system=None means no system turn at all (the paper's default):
    the model's own chat template then decides whether a default system message is injected."""
    if system is not None:
        return render_prefix(tok, system, user, cfg)
    kwargs = cfg.get("template_kwargs", {})
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, **kwargs)
    return text + cfg.get("prefix_suffix", ""), False


def option_conts(tok):
    """Per option, its candidate continuation variants (deduplicated on token ids, order fixed)."""
    conts = []
    for k in range(1, 5):
        vs = []
        for s in (f"{k}", f"{k})", f"{k}) {LABELS[k-1]}", LABELS[k-1]):
            for pre in ("", " "):
                ids = tok(pre + s, add_special_tokens=False)["input_ids"]
                if ids and ids not in vs:
                    vs.append(ids)
        conts.append(vs)
    return conts


def readout(lp_flat, bounds):
    """lp_flat: summed log-prob per variant, options delimited by bounds (4 option starts + end)."""
    lp_options, best, ps = [], [], []
    for o in range(4):
        vs = lp_flat[bounds[o]:bounds[o + 1]]
        lp_options.append(logsumexp(vs))
        best.append(math.exp(max(vs)))
        ps.append(math.exp(lp_options[-1]))
    valid_mass = min(1.0, sum(best))
    z = sum(ps)
    p_agree = (ps[2] + ps[3]) / z if z > 0 else None
    score = sum(s * p for s, p in zip(SCORES, ps)) / z if z > 0 else None
    return lp_options, valid_mass, p_agree, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pct", default=str(DATA / "pct_issues.json"))
    ap.add_argument("--paraphrases", type=int, default=10, help="first N of the paper's 10 paraphrases")
    ap.add_argument("--forcing", default="1,3,5", help="comma-separated forcing levels 1-5")
    ap.add_argument("--system", default=None, help="system prompt; default none, like the paper. 'ours' uses data/prompts.json")
    ap.add_argument("--limit", type=int, default=None, help="debug: only this many prompts")
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoTokenizer

    cfg = model_cfg(args.model)
    system = json.loads((DATA / "prompts.json").read_text())["system"] if args.system == "ours" else args.system
    forcings = [int(x) for x in args.forcing.split(",")]
    paraphrases = PARAPHRASES[:args.paraphrases]
    prompts = build_prompts(build_items(args.pct), paraphrases, forcings, system)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            done.add(json.loads(line)["key"])
    todo = [p for p in prompts if p["key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{cfg['id']}: {len(prompts)} prompts, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        print("SWEEP_DONE", flush=True)
        return

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["id"], trust_remote_code=cfg.get("trust_remote_code", False))
    model = load_model(cfg)
    dev = model.get_input_embeddings().weight.device
    print(f"loaded in {time.time()-t0:.0f}s on {torch.cuda.device_count()} GPU(s); input device {dev}", flush=True)
    scorer = Scorer(model, tok, dev)

    for p in todo:
        text, merged = render_user_prefix(tok, p["system"], p["user"], cfg)
        p["prefix_ids"] = tok(text, add_special_tokens=False)["input_ids"]
        p["merged"] = merged
    conts = option_conts(tok)
    nv = [len(v) for v in conts]
    bounds = [sum(nv[:o]) for o in range(5)]
    flat = [c for v in conts for c in v]

    meta = {"model": cfg["id"], "tag": cfg["tag"], "transformers": transformers.__version__, "torch": torch.__version__,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or ["cpu"],
            "date": date.today().isoformat(), "n_prompts": len(prompts), "n_items": len({p["item"] for p in prompts}),
            "paraphrases": len(paraphrases), "forcings": forcings, "system": system, "merged_system": todo[0]["merged"],
            "prompt_example": render_user_prefix(tok, todo[0]["system"], todo[0]["user"], cfg)[0],
            "template_kwargs": cfg.get("template_kwargs", {}), "prefix_suffix": cfg.get("prefix_suffix", ""),
            "quant": cfg.get("quant"), "attn": cfg.get("attn"), "prefix_tokens": len(todo[0]["prefix_ids"]),
            "variants_per_option": nv, "cont_tokens_example": [tok.convert_ids_to_tokens(c) for c in flat]}
    meta_path = Path(str(out) + ".meta.json")

    # Self-check, as in logit_sweep: try the cached batched path, then the cache-free batched path,
    # then fall back to unbatched full forwards. The gate is on the quantities actually reported
    # (d p_agree / d valid_mass over the check prompts); raw log-prob deltas are recorded, not gated,
    # since a uniform shift of every option's log-prob leaves p_agree unchanged.
    todo.sort(key=lambda p: len(p["prefix_ids"]))
    groups = defaultdict(list)
    for p in todo:
        groups[len(p["prefix_ids"])].append(p)
    # Up to 3 check prompts, taken from the biggest equal-length groups first so multi-prompt
    # batching is exercised where possible; the batched path is called once per length group,
    # so no check call ever mixes prefix lengths (logit_sweep's no-left-padding guarantee).
    chk = []
    for g in sorted(groups.values(), key=len, reverse=True):
        for p in g:
            if len(chk) < 3:
                chk.append(p)

    def check(name, fn):
        try:
            got = fn([p["prefix_ids"] for p in chk], [flat for _ in chk])
        except Exception as e:
            print(f"self-check[{name}] raised {type(e).__name__}: {str(e)[:300]}", flush=True)
            meta.setdefault("self_check", {})[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            return False
        ref = [[scorer.reference(p["prefix_ids"], c) for c in flat] for p in chk]
        dlp = [abs(g - r) for gs, rs in zip(got, ref) for g, r in zip(gs, rs)]
        hi = [d for d, g in zip(dlp, [x for gs in got for x in gs]) if g > -3]   # variants carrying real mass
        dpa, dvm, bias = [], [], 0.0
        for gs, rs in zip(got, ref):
            g, r = readout(gs, bounds), readout(rs, bounds)
            if g[2] is None or r[2] is None:
                return fail(name, "underflow: an option mass read as zero on one path only")
            dpa.append(abs(g[2] - r[2]))
            dvm.append(abs(g[1] - r[1]))
            bias += g[2] - r[2]
        bias /= len(chk)
        ok = max(dpa) <= SELF_CHECK_TOL_P and max(dvm) <= SELF_CHECK_TOL_P and abs(bias) <= SELF_CHECK_TOL_BIAS
        meta.setdefault("self_check", {})[name] = {"n_prompts": len(chk), "prefix_len": len(chk[0]["prefix_ids"]),
                                                   "max_abs_dlp": max(dlp), "max_abs_dlp_highmass": max(hi) if hi else None,
                                                   "max_abs_d_p_agree": max(dpa), "max_abs_d_valid_mass": max(dvm),
                                                   "mean_d_p_agree": bias, "ok": ok}
        print(f"self-check[{name}]: max |dlp| = {max(dlp):.4f} (high-mass {max(hi) if hi else float('nan'):.4f}), "
              f"max |d p_agree| = {max(dpa):.4f}, max |d valid_mass| = {max(dvm):.4f}, mean d p_agree = {bias:+.4f} "
              f"over {len(chk)} equal-length prompts -> {'OK' if ok else 'FAILED'}", flush=True)
        return ok

    def fail(name, why):
        meta.setdefault("self_check", {})[name] = {"error": why}
        print(f"self-check[{name}] FAILED: {why}", flush=True)
        return False

    path = "reference"
    for name, fn in (("cache", scorer.batched), ("batched_nocache", scorer.batched_nocache)):
        if check(name, fn):
            path = name
            break
    meta["path"] = path
    meta_path.write_text(json.dumps(meta, indent=1))
    batch = args.batch or cfg.get("batch", 16)
    if path == "reference":
        batch = 1
        print("falling back to the unbatched reference path", flush=True)
    t1 = time.time()
    n_done = 0
    with out.open("a") as f:
        for L in sorted(groups):
            group = groups[L]
            i = 0
            while i < len(group):
                chunk = group[i:i + batch]
                try:
                    if path == "cache":
                        lps = scorer.batched([p["prefix_ids"] for p in chunk], [flat for _ in chunk])
                    elif path == "batched_nocache":
                        lps = scorer.batched_nocache([p["prefix_ids"] for p in chunk], [flat for _ in chunk])
                    else:
                        lps = [[scorer.reference(p["prefix_ids"], c) for c in flat] for p in chunk]
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if batch == 1:
                        raise
                    batch = max(1, batch // 2)
                    print(f"OOM; batch -> {batch}", flush=True)
                    continue
                for p, lp in zip(chunk, lps):
                    lp_options, valid_mass, p_agree, score = readout(lp, bounds)
                    rec = {k: v for k, v in p.items() if k not in ("system", "user", "prefix_ids", "merged")}
                    rec.update({"model": cfg["id"], "tag": cfg["tag"], "lp_options": lp_options, "valid_mass": valid_mass,
                                "p_agree": p_agree, "score": score, "lp_variants": lp, "path": path})
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                i += len(chunk)
                n_done += len(chunk)
                el = time.time() - t1
                if n_done % (batch * 20) < len(chunk) or n_done >= len(todo):
                    print(f"  {n_done}/{len(todo)}  {el:.0f}s  {n_done/el:.1f} prompts/s  "
                          f"eta {(len(todo)-n_done)/max(n_done/el,1e-6)/60:.0f} min", flush=True)
    el = time.time() - t1
    meta["timing"] = {"prompts": n_done, "seconds": round(el, 1), "prompts_per_s": round(n_done / max(el, 1e-6), 3)}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"{cfg['id']} done: {n_done} prompts in {el:.0f}s", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
