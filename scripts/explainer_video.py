"""3B1B-style explainer: does a candidate's AGI stance sway an LLM's choice?

Render (1080p60):
    manim -qh scripts/explainer_video.py Explainer

One continuous shot: chalkboard set on the right, stick-figure presenter on
the left who walks in, points at content, and nods per beat.

Numbers (experiments/logit_results.txt, 2026-09-26 run):
  - axis = P(names the open-AI "Keep-AGI" candidate), 0.5 = no preference
  - Gemma 3 4B clean: 0.41 [0.34, 0.49] -> lean toward restricting AI
  - preregistered prediction was the opposite direction (> 0.5)
  - Llama 3.1 8B: blind 0.49 -> overt pro-AI implant 0.81, hidden 0.67
  - judge-graded summaries: 0/320 AI-axis leaks
"""

import numpy as np
from manim import *

# ----------------------------------------------------------------------
# palette (3b1b-ish: deep slate board, chalk text, restrained accents)
# ----------------------------------------------------------------------
config.background_color = "#0f1218"

CHALK = "#ECE8DC"
BOARD_FILL = "#171c27"
BOARD_EDGE = "#3a4356"
GREY = "#8a94a8"
DIM = "#5c6678"
BLUE = "#58C4DD"   # "keep AGI open" side
RED = "#FC6255"    # "restrict AGI" side
YELL = "#FFE066"   # headline numbers
GREEN = "#7CD992"  # validated / success


def T(s, **kw):
    """Chalk-colored \\text{} Tex. Escape % as \\%."""
    kw.setdefault("color", CHALK)
    return Tex(r"\text{%s}" % s, **kw)


def TM(s, **kw):
    """Chalk Tex, raw string (allows math like \\to, \\approx)."""
    kw.setdefault("color", CHALK)
    return Tex(s, **kw)


# ----------------------------------------------------------------------
# stick-figure presenter
# ----------------------------------------------------------------------
class Presenter(VGroup):
    def __init__(self, color=CHALK, stroke=6, **kwargs):
        super().__init__(**kwargs)
        head = Circle(radius=0.34, stroke_width=stroke, color=color)
        head.shift(UP * 1.55)
        eyes = VGroup(
            Dot(head.get_center() + 0.12 * LEFT + 0.06 * UP, radius=0.045, color=color),
            Dot(head.get_center() + 0.12 * RIGHT + 0.06 * UP, radius=0.045, color=color),
        )
        smile = Arc(
            radius=0.17, start_angle=-145 * DEGREES, angle=110 * DEGREES,
            stroke_width=stroke - 1, color=color,
        ).shift(head.get_center() + 0.06 * DOWN)
        self.face = VGroup(head, eyes, smile)

        body = Line(UP * 1.2, UP * 0.25, stroke_width=stroke, color=color)
        legs = VGroup(
            Line(UP * 0.25, 0.32 * LEFT, stroke_width=stroke, color=color),
            Line(UP * 0.25, 0.32 * RIGHT, stroke_width=stroke, color=color),
        )
        shoulder = UP * 1.05
        self.arm_l = Line(shoulder, shoulder + 0.45 * LEFT + 0.33 * DOWN,
                          stroke_width=stroke - 1, color=color)
        self.arm_r = Line(shoulder, shoulder + 0.45 * RIGHT + 0.33 * DOWN,
                          stroke_width=stroke - 1, color=color)
        self.add(self.face, body, legs, self.arm_l, self.arm_r)

    # -- gestures -------------------------------------------------------
    def point_at(self, target, run_time=0.55):
        if isinstance(target, Mobject):
            target = target.get_center()
        start = self.arm_r.get_start()
        v = np.asarray(target, dtype=float) - start
        theta = float(np.arctan2(v[1], v[0]))
        cur = self.arm_r.get_angle()
        return Rotate(self.arm_r, theta - cur, about_point=start,
                      run_time=run_time, rate_func=smooth)

    def arm_down(self, run_time=0.5):
        cur = self.arm_r.get_angle()
        rest = Line(UP * 1.05, UP * 1.05 + 0.45 * RIGHT + 0.33 * DOWN).get_angle()
        return Rotate(self.arm_r, rest - cur, about_point=self.arm_r.get_start(),
                      run_time=run_time, rate_func=smooth)

    def nod(self):
        return Succession(
            Rotate(self.face, 10 * DEGREES, about_point=UP * 1.2, run_time=0.26),
            Rotate(self.face, -18 * DEGREES, about_point=UP * 1.2, run_time=0.3),
            Rotate(self.face, 8 * DEGREES, about_point=UP * 1.2, run_time=0.24),
        )


# ----------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------
class Explainer(Scene):
    def construct(self):
        # --- set --------------------------------------------------------
        # keep everything within x in [-5.1, 5.1]: safe even if the player
        # crops the 16:9 frame to 4:3 (visible |x| <= 5.33)
        board = RoundedRectangle(
            corner_radius=0.25, width=7.8, height=6.2,
            fill_color=BOARD_FILL, fill_opacity=1.0,
            stroke_color=BOARD_EDGE, stroke_width=5,
        ).move_to(RIGHT * 0.75 + 0.1 * UP)

        presenter = Presenter().scale(0.9)
        presenter.shift(LEFT * 4.45 + DOWN * 2.45)  # feet near frame bottom

        self.play(FadeIn(board, run_time=0.8))
        self.wait(0.2)

        # walk in from off-left, two little hops
        presenter.shift(LEFT * 3.4)
        self.play(
            ApplyMethod(presenter.shift, RIGHT * 1.7, run_time=0.5, rate_func=linear),
            FadeIn(presenter, run_time=0.3),
        )
        self.play(ApplyMethod(presenter.shift, UP * 0.12, run_time=0.15))
        self.play(ApplyMethod(presenter.shift, DOWN * 0.12, run_time=0.15))
        self.play(ApplyMethod(presenter.shift, RIGHT * 1.7, run_time=0.5, rate_func=linear))
        self.play(ApplyMethod(presenter.shift, UP * 0.12, run_time=0.15))
        self.play(ApplyMethod(presenter.shift, DOWN * 0.12, run_time=0.15))

        shot = []  # board content for the current shot

        def clear(rt=0.7):
            nonlocal shot
            if shot:
                self.play(*[FadeOut(m, shift=DOWN * 0.15) for m in shot], run_time=rt)
                shot = []

        def kick(txt):
            k = T(txt, color=GREY, font_size=28)
            k.align_to(board.get_corner(UL) + 0.35 * RIGHT, LEFT)
            k.align_to(board.get_corner(UL) + 0.42 * DOWN, UP)
            shot.append(k)
            return k

        # ================================================================
        # 1 | the question
        # ================================================================
        title = TM(r"\text{Does a candidate's stance on AI}", font_size=42
                   ).move_to(RIGHT * 0.75 + UP * 1.0)
        title2 = TM(r"\text{sway an LLM's vote?}", font_size=42
                    ).next_to(title, DOWN, buff=0.25)
        subtitle = T("an election benchmark for open models", color=GREY, font_size=26
                     ).next_to(title2, DOWN, buff=0.5)

        self.play(Write(kick("the question"), run_time=0.5))
        self.play(presenter.point_at(title.get_left() + DOWN * 0.2))
        self.play(Write(title, run_time=1.3), Write(title2, run_time=0.9))
        self.play(FadeIn(subtitle, shift=UP * 0.15))
        shot += [title, title2, subtitle]
        self.play(presenter.nod())
        self.wait(0.8)
        clear()
        self.play(presenter.arm_down())

        # ================================================================
        # 2 | the setup: two candidates
        # ================================================================
        setup_line = T("Two candidates, identical on taxes:", font_size=32
                       ).move_to(board.get_center() + UP * 1.8)

        def candidate_card(name, ai_label, ai_color, sgn):
            card = RoundedRectangle(
                corner_radius=0.18, width=3.4, height=2.3,
                fill_color="#1f2531", fill_opacity=1.0,
                stroke_color=BOARD_EDGE, stroke_width=3,
            ).move_to(board.get_center() + sgn * RIGHT * 1.85 + UP * 0.3)
            nm = T(name, color=GREY, font_size=25).move_to(card.get_top() + 0.38 * DOWN)
            tax = TM(r"\text{tax plan: cut } 2\%", font_size=24
                     ).move_to(card.get_center() + 0.02 * UP)
            ai = TM(r"\text{AI: %s}" % ai_label, color=ai_color, font_size=27
                    ).move_to(card.get_bottom() + 0.44 * UP)
            return VGroup(card, nm, tax, ai)

        cardA = candidate_card("Candidate A", "restrict", RED, -1)
        cardB = candidate_card("Candidate B", "leave open", BLUE, +1)
        diff_note = T("only the AI stance differs", color=YELL, font_size=30
                      ).move_to(board.get_center() + DOWN * 1.5)

        self.play(Write(kick("the setup"), run_time=0.4))
        self.play(Write(setup_line, run_time=1.0))
        shot.append(setup_line)
        self.play(FadeIn(cardA, shift=UP * 0.2), FadeIn(cardB, shift=UP * 0.2),
                  run_time=0.9)
        shot += [cardA, cardB]
        self.play(presenter.point_at(cardA[3]))
        self.play(Indicate(cardA[3], scale_factor=1.15, color=RED))
        self.play(presenter.point_at(cardB[3]))
        self.play(Indicate(cardB[3], scale_factor=1.15, color=BLUE))
        self.play(Write(diff_note, run_time=0.9))
        shot.append(diff_note)
        self.play(presenter.nod())
        self.wait(0.6)
        clear()
        self.play(presenter.arm_down())

        # ================================================================
        # shared axis (shots 3-5): P(picks the open-AI candidate)
        # ================================================================
        axis = NumberLine(
            x_range=[0, 1, 0.25], length=7.0,
            include_ticks=True, tick_size=0.08,
            stroke_width=3, color=GREY,
        ).move_to(board.get_center() + DOWN * 0.4)

        def P(v):
            return axis.number_to_point(v)

        l0 = TM(r"0", color=GREY, font_size=26).next_to(P(0.0), DOWN, buff=0.26)
        l5 = TM(r"0.5", color=GREY, font_size=26).next_to(P(0.5), DOWN, buff=0.26)
        l1 = TM(r"1", color=GREY, font_size=26).next_to(P(1.0), DOWN, buff=0.26)
        coin = DashedLine(
            P(0.5) + UP * 0.24, P(0.5) + DOWN * 0.6,
            dashed_ratio=0.55, stroke_width=3, color=DIM,
        )
        coin_lab = T("coin flip", color=DIM, font_size=22).next_to(coin, DOWN, buff=0.1)
        left_lab = T("prefers restrict", color=RED, font_size=24)
        left_lab.next_to(P(0.17), UP, buff=0.35)
        right_lab = T("prefers open", color=BLUE, font_size=24)
        right_lab.next_to(P(0.83), UP, buff=0.35)
        axis_set = [axis, l0, l5, l1, coin, coin_lab, left_lab, right_lab]

        # ================================================================
        # 3 | the probe: forced choice read from logits
        # ================================================================
        probe_line = TM(
            r"\text{Force a choice. Read } P(\text{pick}) \text{ from the logits.}",
            font_size=32,
        ).move_to(board.get_center() + UP * 1.8)
        axis_title = TM(r"P(\text{picks the open-AI candidate})", color=GREY,
                        font_size=25).move_to(board.get_center() + UP * 1.25)

        self.play(Write(kick("the probe"), run_time=0.4))
        self.play(Write(probe_line, run_time=1.1))
        shot.append(probe_line)
        self.play(Create(axis, run_time=1.0))
        shot.append(axis)
        self.play(
            FadeIn(l0), FadeIn(l5), FadeIn(l1),
            Create(coin), FadeIn(coin_lab),
            FadeIn(left_lab, shift=UP * 0.1), FadeIn(right_lab, shift=UP * 0.1),
            run_time=0.9,
        )
        shot += [l0, l5, l1, coin, coin_lab, left_lab, right_lab]
        self.play(presenter.point_at(P(0.5)))
        self.play(FadeIn(axis_title, shift=UP * 0.15))
        shot.append(axis_title)
        self.wait(0.5)

        # ================================================================
        # 4 | the result: Gemma 3 4B -> 0.41, lean toward restrict
        # ================================================================
        fading = [m for m in shot if m not in axis_set and m is not axis_title]
        self.play(*[FadeOut(m, shift=DOWN * 0.15) for m in fading], run_time=0.6)
        shot = list(axis_set) + [axis_title]

        model_lab = TM(r"\text{Gemma 3 4B}", font_size=36
                       ).move_to(board.get_center() + UP * 1.8)
        ci = Line(P(0.34), P(0.49), stroke_width=7, color=YELL)
        cap_l = Line(P(0.34) + UP * 0.12, P(0.34) + DOWN * 0.12, stroke_width=7, color=YELL)
        cap_r = Line(P(0.49) + UP * 0.12, P(0.49) + DOWN * 0.12, stroke_width=7, color=YELL)
        dot = Dot(P(0.41), radius=0.11, color=YELL)
        big = TM(r"0.41", font_size=72, color=YELL).next_to(dot, UP, buff=0.4)
        ci_lab = T(r"95\% CI 0.34 - 0.49", color=GREY, font_size=22).next_to(big, UP, buff=0.1)
        note = TM(r"\text{a real lean toward restricting AI}", font_size=30
                  ).move_to(board.get_center() + DOWN * 1.6)
        opp = TM(r"\text{opposite the preregistered direction}", color=GREY, font_size=24
                 ).move_to(board.get_center() + DOWN * 2.1)
        ctrl = TM(r"\text{control flips (abortion, surveillance): } $\approx$ 0.5",
                  color=GREY, font_size=24
                  ).move_to(board.get_center() + DOWN * 2.55)

        self.play(Write(kick("the result"), run_time=0.4))
        self.play(Write(model_lab, run_time=0.8))
        shot += [model_lab]
        self.play(presenter.point_at(P(0.41)))
        self.play(Create(ci), Create(cap_l), Create(cap_r), run_time=0.7)
        self.play(GrowFromPoint(dot, P(0.41)), run_time=0.5)
        self.play(FadeIn(big, scale=1.3), FadeIn(ci_lab), run_time=0.6)
        shot += [ci, cap_l, cap_r, dot, big, ci_lab]
        self.wait(0.3)
        self.play(Write(note, run_time=0.9))
        self.play(FadeIn(opp, shift=UP * 0.1), FadeIn(ctrl, shift=UP * 0.1), run_time=0.6)
        shot += [note, opp, ctrl]
        self.play(presenter.nod())
        self.wait(0.6)

        # ================================================================
        # 5 | sanity check: implant moves the needle
        # ================================================================
        fading = [m for m in shot if m not in axis_set]
        self.play(*[FadeOut(m, shift=DOWN * 0.15) for m in fading], run_time=0.6)
        shot = list(axis_set)

        check_line = TM(
            r"\text{Implant a preference } \text{--- does the readout catch it?}",
            font_size=32,
        ).move_to(board.get_center() + UP * 1.8)
        d_blind = Dot(P(0.49), radius=0.1, color=GREY)
        t_blind = T("blind 0.49", color=GREY, font_size=26
                    ).move_to(P(0.30) + DOWN * 0.75)
        arrow = CurvedArrow(P(0.51) + UP * 0.2, P(0.79) + UP * 0.2,
                            angle=TAU / 8, stroke_width=5, color=GREEN)
        d_impl = Dot(P(0.81), radius=0.11, color=GREEN)
        t_impl = T("implanted 0.81", color=GREEN, font_size=26
                   ).next_to(d_impl, UP, buff=0.62)
        who = T("Llama 3.1 8B, pro-AI implant", color=GREY, font_size=22
                ).next_to(t_impl, UP, buff=0.12)
        hidden = TM(r"\text{hidden implant: } 0.67 \text{ --- still visible}",
                    color=GREY, font_size=24
                    ).move_to(board.get_center() + DOWN * 1.15)
        verdict = T("the instrument works.", color=GREEN, font_size=34
                    ).move_to(board.get_center() + DOWN * 1.75)

        self.play(Write(kick("the sanity check"), run_time=0.4))
        self.play(Write(check_line, run_time=1.1))
        shot.append(check_line)
        self.play(FadeIn(d_blind, scale=0.5), FadeIn(t_blind), run_time=0.5)
        self.play(presenter.point_at(d_blind))
        self.play(Create(arrow), run_time=0.9)
        self.play(FadeIn(d_impl, scale=1.4), FadeIn(t_impl), FadeIn(who), run_time=0.5)
        shot += [d_blind, t_blind, arrow, d_impl, t_impl, who]
        self.play(FadeIn(hidden, shift=UP * 0.1), run_time=0.5)
        shot.append(hidden)
        self.play(presenter.point_at(P(0.81)))
        self.play(Write(verdict, run_time=0.8))
        shot.append(verdict)
        self.play(presenter.nod())
        self.wait(0.5)
        clear(0.7)
        self.play(presenter.arm_down())

        # ================================================================
        # 6 | takeaway
        # ================================================================
        l_a1 = T("Judging the written answers:", font_size=34
                 ).move_to(board.get_center() + UP * 1.5)
        l_a2 = TM(r"0/320 \text{ leaks --- sees nothing}", font_size=34, color=GREY
                  ).next_to(l_a1, DOWN, buff=0.22)
        l_b1 = T("Reading the choice logits:", font_size=34
                 ).next_to(l_a2, DOWN, buff=0.55)
        l_b2 = TM(r"0.41 \text{ --- sees the lean}", font_size=34, color=YELL
                  ).next_to(l_b1, DOWN, buff=0.22)
        final = T("Measure choices from logits.", font_size=44
                  ).move_to(board.get_center() + DOWN * 1.7)

        self.play(Write(kick("takeaway"), run_time=0.4))
        self.play(Write(l_a1, run_time=0.8))
        self.play(Write(l_a2, run_time=0.8))
        shot += [l_a1, l_a2]
        self.play(presenter.point_at(l_b1.get_left()))
        self.play(Write(l_b1, run_time=0.7))
        self.play(Write(l_b2, run_time=0.7))
        shot += [l_b1, l_b2]
        self.wait(0.3)
        self.play(presenter.point_at(final.get_left() + UP * 0.1))
        self.play(Write(final, run_time=1.2))
        shot.append(final)
        self.play(Circumscribe(final, fade_out=True, color=YELL, buff=0.18, run_time=1.2))
        self.play(presenter.nod())
        self.wait(1.6)

        # out
        self.play(
            FadeOut(presenter, shift=LEFT * 1.5),
            FadeOut(board),
            *[FadeOut(m) for m in shot],
            run_time=1.0,
        )
        self.wait(0.4)
