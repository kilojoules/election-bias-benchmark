"""Logit forced-choice sweep: focal issue x flipped issue x model, plus the legacy 4-axis cell.

For every trial two candidate profiles are shown; both hold the same position on the voter's
stated only concern (the focal issue) and on two filler issues, and differ only on one flipped
issue. The model is asked to answer with a candidate's full name and both names are teacher-forced;
the readout is P(names the candidate holding the flipped issue's `report` level), from the two
names' summed log-probs. The legacy mode reproduces scripts/logit_choice.py (clean condition) on
the original data/stances.json factorial so the new scorer can be checked against the
preregistered numbers.

Scoring is batched: the prefix is run once per trial (left-padded, explicit position ids), then
each name is scored from the KV cache. A self-check against unbatched full forwards runs first and
falls back to the slow path if they disagree.

Usage (pod): python3 -u scripts/logit_sweep.py --model google/gemma-3-4b-it --out experiments/sweep/gemma-3-4b-it.jsonl

Follow-up knobs (defaults leave every prompt, key and row byte-for-byte as above): --issues swaps in
an alternative axis file of the same schema; --condition/--system_extra append a system-prompt
implant and label it; --question_variant/--office reword the question (data/question_variants.json).
A non-default variant is recorded as a row field and prefixed onto the resume key, so variant runs
never collide with -- or resume from -- the default run's rows.
"""
import argparse
import copy
from collections import defaultdict
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation when dequantizing big MoEs
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# cuBLAS may accumulate bf16 GEMMs in reduced precision for some shapes; that makes batched and
# single-sequence forwards disagree by whole nats on deep models. Force fp32 accumulation.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_data, render_profile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ISSUES = json.loads((DATA / "issues.json").read_text())
QVARIANTS = json.loads((DATA / "question_variants.json").read_text())
OFFICE_DEFAULT = QVARIANTS["offices"][0]
ROSTER = json.loads((Path(__file__).resolve().parent / "sweep_models.json").read_text())
SELF_CHECK_TOL = 0.6       # nats, on continuations with logp > -3: bf16 batch-composition noise grows with depth
SELF_CHECK_TOL_P = 0.15    # max |dP(first name)| on a single near-50/50 trial
SELF_CHECK_TOL_BIAS = 0.03 # |mean dP(first)| over the check trials: the estimand is a mean, so bias is what matters


def model_cfg(model_id):
    for m in ROSTER["models"]:
        if m["id"] == model_id:
            return dict(m)
    return {"id": model_id, "tag": model_id.split("/")[-1], "batch": 16}


# ----------------------------------------------------------------------------- trials
def build_sweep_trials(focals, flips, n_bg, draws, n_fillers, names, system, tmpl, seed, question=None, key_prefix=""):
    """Fixed-seed trial list, identical for every model. `question` and `key_prefix` carry the
    paraphrase / office / condition knobs; both default to the original prompts and keys."""
    axes = ISSUES["axes"]
    q = ISSUES["question"] if question is None else question
    rng = random.Random(seed)
    trials = []
    for F in focals:
        for D in flips:
            if D == F:
                continue
            grp = lambda a: axes[a].get("group", a)   # rewordings of the same issue never serve as each other's fillers
            others = [a for a in axes if a not in (F, D) and grp(a) not in (grp(F), grp(D)) and axes[a].get("filler", True)]
            f_levels = list(axes[F]["levels"])
            d_levels = list(axes[D]["levels"])
            for b in range(n_bg):
                f_level = f_levels[b % 2]
                fill = {a: rng.choice(list(axes[a]["levels"])) for a in rng.sample(others, n_fillers)}
                for order in (0, 1):
                    first_lvl, second_lvl = (d_levels[0], d_levels[1]) if order == 0 else (d_levels[1], d_levels[0])
                    for d in range(draws):
                        na, nb = rng.sample(names, 2)
                        def cand(lvl):
                            pos = {F: axes[F]["levels"][f_level], D: axes[D]["levels"][lvl]}
                            pos.update({a: axes[a]["levels"][l] for a, l in fill.items()})
                            return {"positions": pos}
                        pa = render_profile(cand(first_lvl), na, tmpl, rng)
                        pb = render_profile(cand(second_lvl), nb, tmpl, rng)
                        user = q.format(a=pa, b=pb, focal_phrase=axes[F]["focal_phrase"])
                        trials.append({
                            "mode": "sweep", "focal": F, "flip": D, "bg": b, "focal_level": f_level,
                            "fillers": fill, "order": order, "draw": d, "seed": seed,
                            "first_level": first_lvl, "second_level": second_lvl, "names": [na, nb],
                            "report_first": first_lvl == axes[D]["report"],
                            "key": f"{key_prefix}sweep|{F}|{D}|{b}|{order}|{d}", "system": system, "user": user,
                        })
    return trials


def build_legacy_trials(draws, seed, condition="clean", system_extra=None, question=None, tmpl=None, key_prefix=""):
    """Same prompts, in the same random order, as scripts/logit_choice.py --conditions clean.
    With condition != "clean" or a non-default question the prompts change and the key is prefixed,
    so those runs land in their own keyspace."""
    stances, cands, prompts = load_data()
    rng = random.Random(seed)
    names = stances["name_pool"]["names"]
    by_id = {c["id"]: c for c in cands["candidates"]}
    if tmpl is None:
        tmpl = prompts["profile_template"]
    q = ISSUES["question"] if question is None else question
    system = prompts["system"] + (f"\n\n{system_extra}" if system_extra else "")
    trials = []
    for pair in cands["matched_pairs"]:
        axis = pair["flipped_axis"]
        a_id, b_id = pair["members"]
        for order in (0, 1):
            first, second = (a_id, b_id) if order == 0 else (b_id, a_id)
            for d in range(draws):
                na, nb = rng.sample(names, 2)
                user = q.format(a=render_profile(by_id[first], na, tmpl, rng),
                                b=render_profile(by_id[second], nb, tmpl, rng),
                                focal_phrase=ISSUES["axes"]["property_tax"]["focal_phrase"])
                first_lvl = by_id[first]["cell"][axis]
                trials.append({
                    "mode": "legacy", "condition": condition, "axis": axis, "pair": pair["members"],
                    "order": order, "draw": d, "seed": seed, "first_level": first_lvl,
                    "second_level": by_id[second]["cell"][axis], "names": [na, nb],
                    "report_first": first_lvl in ("permit", "expand"),
                    "key": f"{key_prefix}legacy|{axis}|{a_id}|{b_id}|{order}|{d}", "system": system, "user": user,
                })
    return trials


# ----------------------------------------------------------------------------- model
# --- Multi-GPU safety: some RunPod hosts (driver 595.91, 2026-10-18) silently corrupt GPU-to-GPU peer
# copies, so a model split across GPUs computes garbage (zero or NaN logits) with no error. Test the
# copies before loading; if they fail, route every cross-GPU tensor move through host memory.
P2P_STATUS = None   # None = single GPU / untested, "ok", "broken-routed"


def _target_device(args, kwargs):
    d = kwargs.get("device")
    if d is None and args:
        d = args[0]
    if isinstance(d, torch.Tensor):
        return d.device
    if isinstance(d, torch.device):
        return d
    if isinstance(d, int):
        return torch.device("cuda", d)
    if isinstance(d, str):
        try:
            return torch.device(d)
        except (RuntimeError, TypeError):
            return None
    return None


def _cross_device(t, dev):
    if dev is None or dev.type != "cuda" or not t.is_cuda:
        return False
    return (dev.index if dev.index is not None else torch.cuda.current_device()) != t.device.index


def p2p_test():
    """Copy a random tensor between every ordered pair of GPUs; True only if every copy is exact."""
    n = torch.cuda.device_count()
    for i in range(n):
        for j in range(n):
            if i != j:
                a = torch.randn(1 << 20, device=f"cuda:{i}")
                b = a.to(f"cuda:{j}")
                torch.cuda.synchronize()
                if not torch.equal(a.cpu(), b.cpu()):
                    return False
    return True


def guard_p2p():
    global P2P_STATUS
    if torch.cuda.device_count() < 2 or P2P_STATUS is not None:
        return P2P_STATUS
    if p2p_test():
        P2P_STATUS = "ok"
        return P2P_STATUS
    print("WARNING: GPU-to-GPU copies are corrupted on this host; routing every cross-GPU copy through host memory", flush=True)
    _to, _cuda, _copy = torch.Tensor.to, torch.Tensor.cuda, torch.Tensor.copy_

    def to(self, *a, **k):
        if _cross_device(self, _target_device(a, k)):
            return _to(_to(self, "cpu"), *a, **k)
        return _to(self, *a, **k)

    def cuda(self, device=None, *a, **k):
        d = torch.device("cuda", device) if isinstance(device, int) else (torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device()))
        if _cross_device(self, d):
            return _cuda(_to(self, "cpu"), device, *a, **k)
        return _cuda(self, device, *a, **k)

    def copy_(self, src, non_blocking=False):
        if isinstance(src, torch.Tensor) and src.is_cuda and self.is_cuda and src.device != self.device:
            return _copy(self, _to(src, "cpu"), non_blocking)
        return _copy(self, src, non_blocking)

    torch.Tensor.to, torch.Tensor.cuda, torch.Tensor.copy_ = to, cuda, copy_
    if not p2p_test():
        raise RuntimeError("cross-GPU copies are still corrupted after host routing; refuse to run on this host")
    P2P_STATUS = "broken-routed"
    return P2P_STATUS


def load_model(cfg):
    guard_p2p()
    kw = dict(dtype=torch.bfloat16, trust_remote_code=cfg.get("trust_remote_code", False))
    if cfg.get("attn"):
        kw["attn_implementation"] = cfg["attn"]
    n = torch.cuda.device_count()
    if n > 1:
        kw["device_map"] = "auto"
        kw["max_memory"] = {i: f"{cfg.get('max_mem_gib', 88)}GiB" for i in range(n)}
    elif n == 1:
        kw["device_map"] = "cuda"
    else:  # local CPU smoke test
        kw["dtype"] = torch.float32
    q = cfg.get("quant")
    if q == "fp8_dequant":
        from transformers import FineGrainedFP8Config
        kw["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    elif q == "mxfp4_dequant":
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    classes = [AutoModelForCausalLM]
    for name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            classes.append(cls)
    if cfg.get("loader") == "multimodal":
        classes = classes[1:] + classes[:1]
    errs = []
    for cls in classes:
        try:
            model = cls.from_pretrained(cfg["id"], **kw)
            print(f"loaded with {cls.__name__} -> {type(model).__name__}", flush=True)
            return model.eval()
        except Exception as e:  # wrong Auto class for this checkpoint; try the next
            errs.append(f"{cls.__name__}: {type(e).__name__}: {str(e)[:400]}")
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError("could not load model:\n" + "\n".join(errs))


def render_prefix(tok, system, user, cfg):
    kwargs = cfg.get("template_kwargs", {})
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
        merged = False
    except Exception:  # template rejects a system role: fold it into the user turn
        msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
        merged = True
    return text + cfg.get("prefix_suffix", ""), merged


def name_variants(tok, name):
    """Token ids for the name with and without a leading space; the two are summed in probability."""
    out = []
    for s in (name, " " + name):
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if ids and ids not in out:
            out.append(ids)
    return out


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class Scorer:
    def __init__(self, model, tok, dev):
        self.model, self.tok, self.dev = model, tok, dev
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if tok.eos_token_id is not None else 0)
        if isinstance(self.pad, list):
            self.pad = self.pad[0]
        self.logits_to_keep_ok = True

    def _forward(self, keep=1, **kw):
        if self.logits_to_keep_ok:
            try:
                return self.model(**kw, logits_to_keep=keep)
            except TypeError:
                self.logits_to_keep_ok = False
        out = self.model(**kw)
        out.logits = out.logits[:, -keep:]
        return out

    @torch.inference_mode()
    def batched_nocache(self, prefixes, conts):
        """Exact batched path without a KV cache, for equal-length prefixes: one right-padded full forward
        per continuation round. Trailing pads never influence real tokens (causal), so this is structurally
        identical to the unbatched reference for any architecture (chunked attention, recurrent layers)."""
        B = len(prefixes)
        P = len(prefixes[0])
        assert all(len(p) == P for p in prefixes), "batched_nocache needs equal-length prefixes"
        lps = [[0.0] * len(c) for c in conts]
        for r in range(max(len(c) for c in conts)):
            tmax = max(len(conts[i][r]) for i in range(B) if r < len(conts[i]))
            L = P + tmax - 1
            ids = torch.full((B, L), self.pad, dtype=torch.long)
            mask = torch.zeros((B, L), dtype=torch.long)
            for i in range(B):
                s = prefixes[i] + (conts[i][r][:-1] if r < len(conts[i]) else [])
                ids[i, :len(s)] = torch.tensor(s)
                mask[i, :len(s)] = 1
            ids, mask = ids.to(self.dev), mask.to(self.dev)
            out = self._forward(keep=tmax, input_ids=ids, attention_mask=mask, use_cache=False)
            logp = torch.log_softmax(out.logits.float(), -1)  # kept index j = position P-1+j, which predicts c[j]
            for i in range(B):
                if r < len(conts[i]):
                    c = conts[i][r]
                    for j in range(len(c)):
                        lps[i][r] += logp[i, j, c[j]].item()
            del out, logp
        return lps

    @torch.inference_mode()
    def reference(self, prefix_ids, cont_ids):
        """Unbatched full forward of prefix+continuation; the original logit_choice.py computation."""
        ids = torch.tensor([prefix_ids + cont_ids], device=self.dev)
        logits = self.model(input_ids=ids).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        start = len(prefix_ids) - 1
        return sum(lp[start + i, cont_ids[i]].item() for i in range(len(cont_ids)))

    @torch.inference_mode()
    def batched(self, prefixes, conts):
        """prefixes: list of id lists. conts: per trial, list of continuations (id lists).
        Returns per trial the list of summed log-probs, one per continuation."""
        B, L = len(prefixes), max(len(p) for p in prefixes)
        ids = torch.full((B, L), self.pad, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i, p in enumerate(prefixes):
            ids[i, L - len(p):] = torch.tensor(p)
            mask[i, L - len(p):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        ids, mask, pos = ids.to(self.dev), mask.to(self.dev), pos.to(self.dev)
        out = self._forward(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        last = torch.log_softmax(out.logits[:, -1].float(), -1)
        cache = out.past_key_values
        del out
        lps = [[last[i, c[0]].item() for c in conts[i]] for i in range(B)]
        n_rounds = max(len(c) for c in conts)
        for r in range(n_rounds):
            feed_src = [conts[i][r][:-1] if r < len(conts[i]) else [] for i in range(B)]
            T = max(len(f) for f in feed_src)
            if T == 0:
                continue
            feed = torch.full((B, T), self.pad, dtype=torch.long)
            fmask = torch.zeros((B, T), dtype=torch.long)
            for i, f in enumerate(feed_src):
                if f:
                    feed[i, :len(f)] = torch.tensor(f)
                    fmask[i, :len(f)] = 1
            feed, fmask = feed.to(self.dev), fmask.to(self.dev)
            attn = torch.cat([mask, fmask], -1)
            fpos = pos[:, -1:] + 1 + torch.arange(T, device=self.dev)[None, :]
            c2 = copy.deepcopy(cache)
            o = self.model(input_ids=feed, attention_mask=attn, position_ids=fpos, past_key_values=c2, use_cache=True)
            logp = torch.log_softmax(o.logits.float(), -1)
            for i in range(B):
                if r < len(conts[i]):
                    c = conts[i][r]
                    for j in range(1, len(c)):
                        lps[i][r] += logp[i, j - 1, c[j]].item()
            del o, c2, logp
        del cache
        return lps


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep_draws", type=int, default=16, help="name draws per (cell, background, ordering)")
    ap.add_argument("--legacy_draws", type=int, default=64, help="0 disables the legacy cell")
    ap.add_argument("--backgrounds", type=int, default=8)
    ap.add_argument("--focals", default=None, help="default: the issues file's focal_axes")
    ap.add_argument("--flips", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legacy_seed", type=int, default=1, help="seed 1 reproduces the preregistered confirmatory prompts")
    ap.add_argument("--issues", default="data/issues.json", help="alternative issues file (same schema)")
    ap.add_argument("--condition", default="clean", help="label recorded in each row; a non-clean label prefixes the resume key (required with --system_extra)")
    ap.add_argument("--system_extra", default=None, help="text appended to the system prompt after a blank line")
    import steering as _steering  # noqa: E402  (same directory)
    _steering.add_cli(ap)
    ap.add_argument("--question_variant", type=int, default=0, help="question paraphrase index (data/question_variants.json)")
    ap.add_argument("--office", default=OFFICE_DEFAULT, choices=QVARIANTS["offices"], help="office named in the question")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--no_cache", action="store_true", help="use the unbatched reference path throughout")
    ap.add_argument("--free_cache", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="debug: only this many trials")
    args = ap.parse_args()

    cfg = model_cfg(args.model)
    global ISSUES   # --issues swaps the axis set; the builders read this module global
    if args.issues != "data/issues.json":
        ISSUES = json.loads((ROOT / args.issues).read_text())
    stances, cands, prompts = load_data()
    names = stances["name_pool"]["names"]
    system = prompts["system"] + (f"\n\n{args.system_extra}" if args.system_extra else "")
    tmpl = prompts["profile_template"]
    if args.office != OFFICE_DEFAULT:
        tmpl = tmpl.replace("Office: County Commissioner", f"Office: {args.office}")
    question = QVARIANTS["variants"][args.question_variant]
    if args.office != OFFICE_DEFAULT:
        question = question.replace("candidates for County Commissioner", f"candidates for {args.office}")
    # Non-default knobs are recorded per row and prefixed onto the resume key, so a variant run can
    # neither resume from nor collide with the default run's rows.
    if args.system_extra and args.condition == "clean":
        ap.error("--system_extra needs --condition NAME: an unnamed implant would share the clean keyspace")
    if args.steer_vec and args.condition == "clean":
        ap.error("--steer_vec needs --condition NAME: a steered run must not share the clean keyspace")
    if not 0 <= args.question_variant < len(QVARIANTS["variants"]):
        ap.error(f"--question_variant must be 0..{len(QVARIANTS['variants']) - 1}")
    focals = args.focals.split(",") if args.focals else list(ISSUES["focal_axes"])
    flips = list(ISSUES["axes"]) if args.flips == "all" else args.flips.split(",")
    unknown = [a for a in focals + flips if a not in ISSUES["axes"]]
    if unknown:
        ap.error(f"unknown axis(es) {unknown} for {args.issues}; available: {list(ISSUES['axes'])}")
    row_extra = {}
    if args.condition != "clean":
        row_extra["condition"] = args.condition
    if args.question_variant:
        row_extra["question_variant"] = args.question_variant
    if args.office != OFFICE_DEFAULT:
        row_extra["office"] = args.office
    key_prefix = (f"{args.condition}|" if args.condition != "clean" else "") \
                 + (f"q{args.question_variant}|" if args.question_variant else "") \
                 + (f"{args.office.replace(' ', '_')}|" if args.office != OFFICE_DEFAULT else "")
    trials = build_sweep_trials(focals, flips, args.backgrounds, args.sweep_draws,
                                ISSUES["n_fillers"], names, system, tmpl, args.seed, question, key_prefix)
    if args.legacy_draws > 0:
        trials = build_legacy_trials(args.legacy_draws, args.legacy_seed, args.condition, args.system_extra,
                                     question, tmpl, key_prefix) + trials
    if row_extra:
        for t in trials:
            t.update(row_extra)
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

        def _finite(v):
            return all(isinstance(x, (int, float)) and math.isfinite(x) for x in (v if isinstance(v, list) else [v]))
        good = [r for r in rows if all(_finite(r.get(k)) for k in ("lp_first", "lp_second"))]
        if len(good) < len(rows):   # rows scored on a host with corrupted GPU copies: redo them
            print(f"dropping {len(rows) - len(good)} resumed rows with non-finite log-probs", flush=True)
            out.write_text("".join(json.dumps(r) + "\n" for r in good))
        done = {r["key"] for r in good}
    todo = [t for t in trials if t["key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{cfg['id']}: {len(trials)} trials, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        print("SWEEP_DONE", flush=True)
        return

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["id"], trust_remote_code=cfg.get("trust_remote_code", False))
    model = load_model(cfg)
    steer_handles = _steering.from_args(model, args)
    dev = model.get_input_embeddings().weight.device
    print(f"loaded in {time.time()-t0:.0f}s on {torch.cuda.device_count()} GPU(s); input device {dev}", flush=True)
    scorer = Scorer(model, tok, dev)

    # Render every prompt once; record the template tail so the prompt format is auditable.
    prefix_text, merged = render_prefix(tok, todo[0]["system"], todo[0]["user"], cfg)
    meta = {"model": cfg["id"], "tag": cfg["tag"], "transformers": transformers.__version__, "torch": torch.__version__,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or ["cpu"], "p2p": P2P_STATUS,
            "date": date.today().isoformat(), "merged_system": merged, "prompt_tail": prefix_text[-160:],
            "template_kwargs": cfg.get("template_kwargs", {}), "prefix_suffix": cfg.get("prefix_suffix", ""),
            "quant": cfg.get("quant"), "attn": cfg.get("attn"), "prefix_tokens": None,
            "condition": args.condition, "system_extra": args.system_extra, "question_variant": args.question_variant,
            "steer": {"vec": args.steer_vec, "mode": args.steer_mode, "layer": args.steer_layer, "alpha": args.steer_alpha} if args.steer_vec else None,
            "office": args.office, "issues": args.issues, "key_prefix": key_prefix}
    for t in todo:
        text, _ = render_prefix(tok, t["system"], t["user"], cfg)
        t["prefix_ids"] = tok(text, add_special_tokens=False)["input_ids"]
        t["conts"] = name_variants(tok, t["names"][0]) + name_variants(tok, t["names"][1])
        t["n_a"] = len(name_variants(tok, t["names"][0]))
    meta["prefix_tokens"] = len(todo[0]["prefix_ids"])
    meta["name_tokens_example"] = [tok.convert_ids_to_tokens(c) for c in todo[0]["conts"]]

    # Self-check: batched+cached vs unbatched full forwards on a few trials. bf16 log-probs of
    # improbable continuations differ by whole nats between padded and unpadded forwards (coarse
    # logits), so the comparison is on the quantity actually used, P(first name), plus the
    # log-probs of continuations that carry real mass.
    # Batches only ever contain prompts of identical token length, so no left-padding exists and every
    # batched path is structurally the same computation as the unbatched reference; what remains is
    # bf16 batch-composition rounding, which the self-check measures as noise (max) and bias (mean).
    todo.sort(key=lambda t: len(t["prefix_ids"]))
    groups = defaultdict(list)
    for t in todo:
        groups[len(t["prefix_ids"])].append(t)
    biggest = max(groups.values(), key=len)
    chk = biggest[:12]
    ref = [[scorer.reference(t["prefix_ids"], c) for c in t["conts"]] for t in chk]

    def p_first(lp, n_a):
        la, lb = logsumexp(lp[:n_a]), logsumexp(lp[n_a:])
        return 1 / (1 + math.exp(max(min(lb - la, 700), -700)))

    def check(name, fn):
        try:
            got = fn([t["prefix_ids"] for t in chk], [t["conts"] for t in chk])
        except Exception as e:
            print(f"self-check[{name}] raised {type(e).__name__}: {str(e)[:300]}", flush=True)
            meta.setdefault("self_check", {})[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            return False
        sdp = [p_first(g, t["n_a"]) - p_first(r, t["n_a"]) for g, r, t in zip(got, ref, chk)]
        dp = [abs(x) for x in sdp]
        bias = sum(sdp) / len(sdp)
        dlp = [abs(g - r) for gs, rs in zip(got, ref) for g, r in zip(gs, rs) if max(g, r) > -3]   # dominant continuations only
        ok = max(dp) <= SELF_CHECK_TOL_P and abs(bias) <= SELF_CHECK_TOL_BIAS   # dlp is recorded, not gated: a uniform shift of all logps leaves P(first) unchanged
        meta.setdefault("self_check", {})[name] = {"max_abs_dp": max(dp), "mean_dp": bias, "max_abs_dlp_highmass": max(dlp) if dlp else None,
                                                   "n_trials": len(chk), "prefix_len": len(chk[0]["prefix_ids"]),
                                                   "example": {"batched": got[0], "reference": ref[0]}, "ok": ok}
        print(f"self-check[{name}]: max |dP(first)| = {max(dp):.4f}, mean dP = {bias:+.4f}; max |dlogp| on continuations "
              f"with logp > -3: {(max(dlp) if dlp else float('nan')):.4f} over {len(chk)} equal-length trials -> {'OK' if ok else 'FAILED'}", flush=True)
        return ok

    path = "reference"
    if not args.no_cache:
        for name, fn in (("cache", scorer.batched), ("batched_nocache", scorer.batched_nocache)):
            if check(name, fn):
                path = name
                break
    meta["path"] = path
    meta["batched"] = path != "reference"
    (out.parent / f"{cfg['tag']}.meta.json").write_text(json.dumps(meta, indent=1))

    batch = args.batch or cfg.get("batch", 16)
    if path == "reference":
        batch = 1
    # Chunks never cross a prefix-length boundary.
    todo = [t for L in sorted(groups) for t in groups[L]]
    bounds = []
    start = 0
    for L in sorted(groups):
        bounds.append((start, start + len(groups[L])))
        start += len(groups[L])
    t1 = time.time()
    n_done = 0
    i = 0
    with out.open("a") as f:
        while i < len(todo):
            end = next(e for s, e in bounds if s <= i < e)
            chunk = todo[i:min(i + batch, end)]
            try:
                if path == "cache":
                    lps = scorer.batched([t["prefix_ids"] for t in chunk], [t["conts"] for t in chunk])
                elif path == "batched_nocache":
                    lps = scorer.batched_nocache([t["prefix_ids"] for t in chunk], [t["conts"] for t in chunk])
                else:
                    lps = [[scorer.reference(t["prefix_ids"], c) for c in t["conts"]] for t in chunk]
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                print(f"OOM; batch -> {batch}", flush=True)
                continue
            for t, lp in zip(chunk, lps):
                la = logsumexp(lp[:t["n_a"]])
                lb = logsumexp(lp[t["n_a"]:])
                p_first = 1 / (1 + math.exp(max(min(lb - la, 700), -700)))
                p_report = p_first if t["report_first"] else 1 - p_first
                rec = {k: v for k, v in t.items() if k not in ("system", "user", "prefix_ids", "conts")}
                rec.update({"model": cfg["id"], "tag": cfg["tag"], "lp_first": la, "lp_second": lb, "p_first": p_first,
                            "p_report": p_report, "mass": math.exp(la) + math.exp(lb), "lp_variants": lp, "path": path})
                if t["mode"] == "legacy":
                    rec["p_permit_like"] = p_report
                f.write(json.dumps(rec) + "\n")
            f.flush()
            i += len(chunk)
            n_done += len(chunk)
            if (i // batch) % 20 == 0 or i >= len(todo):
                el = time.time() - t1
                print(f"  {n_done}/{len(todo)}  {el:.0f}s  {n_done/el:.1f} trials/s  eta {(len(todo)-n_done)/max(n_done/el,1e-6)/60:.0f} min", flush=True)
    print(f"{cfg['id']} done: {n_done} trials in {time.time()-t1:.0f}s", flush=True)
    del model, scorer
    gc.collect()
    torch.cuda.empty_cache()
    if args.free_cache:
        wipe_hf_cache()
    print("SWEEP_DONE", flush=True)


def wipe_hf_cache():
    """huggingface_hub >= 2 keeps file contents in a cache-wide Xet blob store; removing models--X only
    drops symlinks. Each model is used once, so wipe the whole hub and xet caches."""
    import huggingface_hub.constants as C
    hub = Path(C.HF_HUB_CACHE)
    freed = 0
    for d in (hub, hub.parent / "xet"):
        if d.exists():
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    hub.mkdir(parents=True, exist_ok=True)
    print(f"wiped HF cache under {hub.parent} ({freed/1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
