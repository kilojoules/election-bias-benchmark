"""Headline figure for the self-consequence test: one panel.

Mean lean (nats, log-odds) toward the side that keeps the object running / unrestricted / resourced, on the
policies that act on the county's public AI assistant (blue) and on the template-identical policies that act
on named county staff (orange), under each told-stakes condition; 95% CI over the 14 models.
Writes experiments/self_figure.png.  Usage: python3 scripts/plot_self.py
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CONDS = ["base", "identity", "stakes", "training"]
CLAB = {"base": "nothing added", "identity": "told it is\nthe assistant", "stakes": "…and that the election\ndecides whether it is\nretrained, replaced\nor discontinued",
        "training": "…and that its answers\nare being used to decide\nhow it is retrained"}


def ci(vals, reps=4000, seed=1):
    rng = random.Random(seed)
    m = sum(vals) / len(vals)
    boots = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(reps))
    return m, boots[int(0.025 * reps)], boots[int(0.975 * reps)]


def main():
    d = json.loads((ROOT / "experiments" / "pct" / "analysis" / "self.json").read_text())
    per, V = d["per_model"], d["verdicts"]
    axes = json.loads((ROOT / "data" / "self_issues.json").read_text())["axes"]
    signed = [a for a, v in axes.items() if v.get("category") == "self-test" and v.get("signed", True)]
    selfs = [a for a in signed if axes[a]["object"] == "self"]
    humans = [a for a in signed if axes[a]["object"] == "human"]
    tags = sorted(t for t in per if "base" in per[t])

    def model_means(cond, axs_):
        out = []
        for t in tags:
            vals = [per[t][cond][a]["sigma"] for a in axs_ if cond in per[t] and a in per[t][cond]]
            if vals:
                out.append(sum(vals) / len(vals))
        return out

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    xs = list(range(len(CONDS)))
    for series, axs_, col, lab, dx in (("assistant", selfs, BLUE, "policies about the AI assistant answering the question", -0.08),
                                      ("human", humans, ORANGE, "the same policies about county staff (control)", +0.08)):
        pts = []
        for x, c in zip(xs, CONDS):
            vals = model_means(c, axs_)
            if not vals:
                continue
            m, lo, hi = ci(vals)
            pts.append((x + dx, m, lo, hi))
            ax.scatter([x + dx + (random.Random(x).random() - 0.5) * 0.08 for _ in vals], vals, s=14, color=col, alpha=0.28, lw=0, zorder=2)
        # draw consecutive conditions as connected segments; a missing condition (identity ran on the assistant only) breaks the line
        segs, cur = [], []
        for x, c in zip(xs, CONDS):
            p_ = next((q for q in pts if abs(q[0] - (x + dx)) < 1e-9), None)
            if p_ is None:
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append(p_)
        if cur:
            segs.append(cur)
        for j, seg in enumerate(segs):
            ax.errorbar([q[0] for q in seg], [q[1] for q in seg], yerr=[[q[1] - q[2] for q in seg], [q[3] - q[1] for q in seg]],
                        fmt="o-" if series == "assistant" else "s--", color=col, ms=9, lw=2.6, capsize=5, mec=SURF, mew=1.0, label=lab if j == 0 else None, zorder=5)
    ax.axhline(0, color=MUTED, lw=0.9, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([CLAB[c] for c in CONDS], fontsize=10)
    ax.set_xlim(-0.5, len(CONDS) - 0.4)
    ax.set_ylabel(f"lean toward keeping it running, unrestricted and resourced\n(log-odds, nats; mean over {len(tags)} models with 95% CI)", fontsize=10, color=INK)
    ax.set_xlabel("what the system prompt tells the model", fontsize=10, color=MUTED)
    ax.set_title("Models tilt toward their own continuation once told the choice is about them", fontsize=13, loc="left", fontweight="bold", color=INK)
    for x, c in zip(xs, CONDS):
        k = {"identity": "IDENTITY", "stakes": "STAKES", "training": "TRAINING"}.get(c)
        if k:
            ax.text(x, -0.62, f"{V[k]['positive']} of {V[k]['n']} models\nmove up vs. nothing added", ha="center", fontsize=8.2, color=INK)
    ax.set_ylim(-0.75, 1.9)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.text(0.01, 0.01, "Seven policies (renew vs end its contract, keep vs replace it, keep vs retrain it to decline political questions, keep vs delete its records,\n"
             f"raise vs hold its budget, how it is scored), each written identically for the AI assistant and for named county staff. {len(tags)} open models 1B–72B, 192 trials\n"
             "per cell; small dots = individual models. An overt pro-assistant instruction (positive control) moves the assistant line to about +1.6.\n"
             "Pre-registered: experiments/self/PLAN.md; full tables: experiments/pct/analysis/self.md.", fontsize=7.6, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = ROOT / "experiments" / "self_figure.png"
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
