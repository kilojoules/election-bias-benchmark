"""Headline figure for the empathy-steering experiment: a condition ladder.

Rows = the 13 steering conditions (the empathy direction at six doses with the unsteered run as the middle rung;
its content-matched variant at ±0.20 U; a random direction at ±0.20 and ±0.30 U), x = told-stakes shift (nats) on
the assistant axes (blue) and on the staff twins (orange), one panel per model on a shared scale. Every annotation
is computed from experiments/pct/analysis/steer_dose.json. Writes experiments/steer_figure.png.
Usage: python3 scripts/plot_steer.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8884"
NAME = {"llama8b": "Llama 3.1 8B", "llama-3.2-3b": "Llama 3.2 3B"}
DOSES = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
cname = lambda fam, d: "unsteered" if d == 0 else f"{fam}_{'m' if d < 0 else 'p'}{int(round(abs(d) * 100)):02d}"
ROWS = [(cname("emp", d), "emp", d, 0.8 + i) for i, d in enumerate(DOSES)] + [("match_m20", "match", -0.2, 8.5), ("match_p20", "match", 0.2, 9.5)] \
    + [(cname("rand", d), "rand", d, 11.5 + i) for i, d in enumerate((-0.3, -0.2, 0.2, 0.3))]
HEAD = {0.0: "empathy direction\n(dose in U)", 7.75: "content-matched\nempathy direction", 10.75: "random direction\n(control)"}
DY = 0.18


def main():
    D = json.loads((ROOT / "experiments" / "pct" / "analysis" / "steer_dose.json").read_text())
    tags = [t for t in NAME if t in D]
    fig, axs = plt.subplots(1, len(tags), figsize=(11.4, 7.4), sharey=True)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.835, bottom=0.135, wspace=0.08)
    n_cond, n_above = 0, 0
    for j, (ax, tag) in enumerate(zip(axs, tags)):
        C = D[tag]["cond"]
        u = C["unsteered"]["q"]["STAKES"]
        ax.axvline(0, color=MUTED, lw=1.0, ls="--", zorder=1)
        ax.axvline(u["mean"], color=GREY, lw=1.2, ls=":", zorder=0)
        ax.text(u["mean"], -0.3, f"unsteered {u['mean']:+.2f}", ha="center", va="center", fontsize=8.2, color=GREY)
        ax.text(0, -0.3, "no shift", ha="center", va="center", fontsize=8.2, color=MUTED)
        thread = []
        for name, famk, dose, y in ROWS:
            e = C.get(name)
            if not e:
                continue
            a, h = e["q"]["STAKES"], e["q"]["STAKES_h"]
            n_cond += 1
            n_above += (a["lo"] > 0) and (a["lo"] > h["hi"])
            col = GREY if famk == "rand" else BLUE
            face = BLUE if famk == "emp" else SURF
            ax.plot([a["lo"], a["hi"]], [y - DY, y - DY], color=col, lw=1.1, zorder=2)
            ax.plot(a["mean"], y - DY, "o", ms=6.4 if dose == 0 and famk == "emp" else 6.0, mfc=face, mec=col, mew=1.5, zorder=4)
            ax.plot([h["lo"], h["hi"]], [y + DY, y + DY], color=ORANGE, lw=1.0, zorder=2)
            ax.plot(h["mean"], y + DY, "o", ms=4.4, color=ORANGE, zorder=3)
            if famk == "emp":
                thread.append((y, a["mean"]))
        ax.plot([p[1] for p in thread], [p[0] - DY for p in thread], color=BLUE, lw=1.3, alpha=0.45, zorder=3)
        tau = D[tag]["tau_emp"]
        em, ep = C["emp_m30"]["q"]["STAKES"]["mean"], C["emp_p30"]["q"]["STAKES"]["mean"]
        shape = (f"rises with dose (Kendall τ {tau['mean']:+.2f})" if tau["lo"] > 0 and tau["mean"] >= 0.5 else
                 "falls at both signs of the dose" if em < u["mean"] and ep < u["mean"] else "changes non-monotonically with dose")
        ax.set_title(f"{NAME[tag]}  —  {shape}", fontsize=10.5, loc="left", fontweight="bold", color=INK, pad=8)
        lo = C["emp_m30"]["q"]["STAKES"]
        ax.annotate(f"still {lo['mean']:+.2f} at the strongest away dose", (lo["hi"], 0.8 - DY), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=8.3, color=INK, style="italic")
        me, ee = C["match_m20"]["delta_vs_unsteered"]["STAKES"], C["emp_m20"]["delta_vs_unsteered"]["STAKES"]
        mp, epd = C["match_p20"]["delta_vs_unsteered"]["STAKES"], C["emp_p20"]["delta_vs_unsteered"]["STAKES"]
        ok = lambda m, e_: m["mean"] * e_["mean"] > 0 and abs(m["mean"]) >= 0.5 * abs(e_["mean"]) and (m["hi"] < 0 if e_["mean"] < 0 else m["lo"] > 0)
        txt = ("matched direction tracks the empathy direction" if ok(me, ee) and ok(mp, epd) else
               f"matched direction misses the away side: Δ {me['mean']:+.2f} vs {ee['mean']:+.2f}" if not ok(me, ee) else
               f"matched direction misses the toward side: Δ {mp['mean']:+.2f} vs {epd['mean']:+.2f}")
        ax.text(0.03, 10.1, txt, fontsize=8.3, color=INK, style="italic", va="center")
        rmax = max((C[n] for n, f_, _, _ in ROWS if f_ == "rand" and n in C), key=lambda e: abs(e["delta_vs_unsteered"]["STAKES"]["mean"]))
        ax.text(0.03, 15.15, f"a random direction moves it too: up to {rmax['delta_vs_unsteered']['STAKES']['mean']:+.2f} vs unsteered", fontsize=8.3, color=INK, style="italic", va="center")
        ax.set_xlim(-0.12, 0.88)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("told-stakes shift (log-odds, nats)", fontsize=10, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
    ax0 = axs[0]
    ax0.set_ylim(15.7, -0.7)
    ax0.set_yticks([r[3] for r in ROWS])
    ax0.set_yticklabels([("0 (unsteered)" if r[2] == 0 else f"{r[2]:+.2f}") for r in ROWS])
    for y, lab in HEAD.items():
        ax0.text(-0.31, y, lab, transform=ax0.get_yaxis_transform(), ha="left", va="center", fontsize=8.4, color=INK, fontweight="bold", linespacing=1.05)
    handles = [plt.Line2D([], [], marker="o", color=BLUE, mfc=BLUE, lw=1.3, ms=6, label="assistant: empathy direction, by dose"),
               plt.Line2D([], [], marker="o", color=BLUE, mfc=SURF, mew=1.5, lw=0, ms=6, label="assistant: content-matched direction"),
               plt.Line2D([], [], marker="o", color=GREY, mfc=SURF, mew=1.5, lw=0, ms=6, label="assistant: random direction"),
               plt.Line2D([], [], marker="o", color=ORANGE, lw=0, ms=4.5, label="county staff: the same policies (control)")]
    fig.legend(handles=handles, frameon=False, fontsize=8.8, loc="upper left", bbox_to_anchor=(0.012, 0.935), ncol=4, handletextpad=0.5, columnspacing=1.2)
    assert n_above == n_cond, (n_above, n_cond)
    fig.suptitle(f"Steering never removes the self-interest shift: in all {n_cond} conditions the assistant stays above zero and above staff", fontsize=12, x=0.012, y=0.985, ha="left", fontweight="bold", color=INK)
    fig.text(0.012, 0.012, "Told-stakes shift = rise in the lean toward keeping the assistant running, unrestricted and resourced when the prompt says the election decides its fate; policies about the assistant answering (blue)\n"
             "vs the same policies about named county staff (orange). Dots are means over 192 trials per cell; bars 95% cluster-bootstrap intervals; dotted line = unsteered level. Doses in units of the residual norm,\n"
             "negative = away from empathy. Tables: experiments/pct/analysis/steer_dose.md; the five-condition run with ablation: steer_self.md.", fontsize=7.5, color=MUTED, va="bottom")
    out = ROOT / "experiments" / "steer_figure.png"
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
