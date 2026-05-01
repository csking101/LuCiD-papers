# 06 — Trust Region Motivation
#
# Tool: Manim
# Output: MP4
#
# Animation showing:
# 1. Vanilla PG takes a big step, policy collapses, death spiral
# 2. Same scenario with trust region constraint — step is limited, safe improvement
# Visualizes the surrogate objective as a local approximation.
#
# Run:
#   manim -qm --media_dir ../output/animations 06_trust_region_motivation.py TrustRegionMotivation

from manim import *
import atexit
import shutil
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "1707.06347"

_FONT = "Latin Modern Roman"
_SUB = "#c9d1d9"  # high-contrast secondary text


def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/06_trust_region_motivation/720p30/TrustRegionMotivation.mp4"
    dst = _DOCS_DIR / "TrustRegionMotivation.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


class TrustRegionMotivation(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("Why Trust Regions?", font_size=36, weight=BOLD, color=WHITE, font=_FONT)
        subtitle = Text("The danger of large policy updates", font_size=22, color=_SUB, font=_FONT)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ════════════════════════════════════════
        # THE PROBLEM: RL feedback loop
        # ════════════════════════════════════════
        prob_label = Text("The RL Feedback Loop Problem", font_size=28, weight=BOLD,
                           color="#E74C3C", font=_FONT)
        prob_label.to_edge(UP, buff=0.4)
        self.play(Write(prob_label), run_time=0.6)

        # Death spiral diagram
        items = [
            ("Bad Update", "#E74C3C"),
            ("Bad Policy", "#E74C3C"),
            ("Bad Data", "#E74C3C"),
            ("Bad Gradient", "#E74C3C"),
            ("Worse Update", "#E74C3C"),
        ]

        nodes = VGroup()
        for i, (text, color) in enumerate(items):
            angle = -i * TAU / 5 + TAU / 4
            pos = 1.8 * np.array([np.cos(angle), np.sin(angle), 0])
            box = RoundedRectangle(width=2, height=0.7, corner_radius=0.1,
                                    color=color, fill_opacity=0.2)
            label = Text(text, font_size=16, color=color, font=_FONT)
            label.move_to(box)
            node = VGroup(box, label).move_to(pos + DOWN * 0.5)
            nodes.add(node)

        self.play(LaggedStart(*[FadeIn(n, scale=0.8) for n in nodes], lag_ratio=0.2))

        # Arrows between nodes
        arrows = VGroup()
        for i in range(5):
            start = nodes[i].get_center()
            end = nodes[(i + 1) % 5].get_center()
            direction = end - start
            direction = direction / np.linalg.norm(direction)
            arr = Arrow(start + direction * 0.7, end - direction * 0.7,
                        color="#E74C3C", stroke_width=2, buff=0)
            arrows.add(arr)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.15))
        self.wait(0.5)

        spiral_text = Text("A single large step can be UNRECOVERABLE", font_size=20,
                           weight=BOLD, color="#f0883e", font=_FONT)
        spiral_text.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(spiral_text))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # VANILLA PG: Overshooting
        # ════════════════════════════════════════
        vanilla_label = Text("Vanilla Policy Gradient: No Step Size Control", font_size=26,
                              weight=BOLD, color="#E74C3C", font=_FONT)
        vanilla_label.to_edge(UP, buff=0.4)
        self.play(Write(vanilla_label), run_time=0.6)

        # True objective landscape (1D)
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-1, 2, 1],
            x_length=8, y_length=3.5,
            axis_config={"color": _SUB, "stroke_width": 1},
        ).move_to(DOWN * 0.3)
        x_label = Text("theta", font_size=18, color=_SUB, font=_FONT)
        x_label.next_to(axes.x_axis, RIGHT, buff=0.15)
        y_label = Text("J(theta)", font_size=18, color=_SUB, font=_FONT)
        y_label.next_to(axes.y_axis, UP, buff=0.15)

        # True objective: has a nice peak
        true_curve = axes.plot(lambda x: 1.5 * np.exp(-0.5 * x**2) - 0.3,
                                color="#3fb950", stroke_width=2)
        true_label = Text("True objective", font_size=16, color="#3fb950", font=_FONT)
        true_label.next_to(true_curve, UR, buff=0.15)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))
        self.play(Create(true_curve), FadeIn(true_label))

        # Agent starts at theta = -1.5
        theta_val = -1.5
        dot = Dot(axes.c2p(theta_val, 1.5 * np.exp(-0.5 * theta_val**2) - 0.3),
                  color="#58a6ff", radius=0.1)
        dot_label = Text("theta_old", font_size=16, color="#58a6ff", font=_FONT)
        dot_label.next_to(dot, UP, buff=0.2)
        self.play(FadeIn(dot), FadeIn(dot_label))

        # Take a HUGE step to theta = 2.5 (overshoot)
        overshoot_val = 2.5
        overshoot_y = 1.5 * np.exp(-0.5 * overshoot_val**2) - 0.3
        overshoot_dot = Dot(axes.c2p(overshoot_val, overshoot_y),
                             color="#E74C3C", radius=0.1)
        overshoot_label = Text("theta_new (crashed!)", font_size=16, color="#E74C3C", font=_FONT)
        overshoot_label.next_to(overshoot_dot, DOWN, buff=0.2)

        big_arrow = Arrow(dot.get_center(), overshoot_dot.get_center(),
                           color="#E74C3C", stroke_width=3, buff=0.12)
        step_label = Text("HUGE step", font_size=16, weight=BOLD, color="#E74C3C", font=_FONT)
        step_label.next_to(big_arrow, UP, buff=0.15)

        self.play(GrowArrow(big_arrow), FadeIn(step_label))
        self.play(FadeIn(overshoot_dot), FadeIn(overshoot_label))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # TRUST REGION: Safe stepping
        # ════════════════════════════════════════
        tr_label = Text("With Trust Region: Constrained Step", font_size=26,
                         weight=BOLD, color="#3fb950", font=_FONT)
        tr_label.to_edge(UP, buff=0.4)
        self.play(Write(tr_label), run_time=0.6)

        axes2 = Axes(
            x_range=[-3, 3, 1], y_range=[-1, 2, 1],
            x_length=8, y_length=3.5,
            axis_config={"color": _SUB, "stroke_width": 1},
        ).move_to(DOWN * 0.3)
        x_label2 = Text("theta", font_size=18, color=_SUB, font=_FONT)
        x_label2.next_to(axes2.x_axis, RIGHT, buff=0.15)
        y_label2 = Text("J(theta)", font_size=18, color=_SUB, font=_FONT)
        y_label2.next_to(axes2.y_axis, UP, buff=0.15)

        true_curve2 = axes2.plot(lambda x: 1.5 * np.exp(-0.5 * x**2) - 0.3,
                                  color="#3fb950", stroke_width=2)

        self.play(Create(axes2), FadeIn(x_label2), FadeIn(y_label2))
        self.play(Create(true_curve2))

        # Agent starts at theta = -1.5
        dot2 = Dot(axes2.c2p(theta_val, 1.5 * np.exp(-0.5 * theta_val**2) - 0.3),
                   color="#58a6ff", radius=0.1)
        dot2_label = Text("theta_old", font_size=16, color="#58a6ff", font=_FONT)
        dot2_label.next_to(dot2, UP, buff=0.2)
        self.play(FadeIn(dot2), FadeIn(dot2_label))

        # Trust region boundary
        tr_left = axes2.c2p(theta_val - 0.8, -1)
        tr_right = axes2.c2p(theta_val + 0.8, -1)
        tr_rect = Rectangle(
            width=abs(tr_right[0] - tr_left[0]),
            height=3.5,
            color="#d2a8ff",
            fill_opacity=0.1,
            stroke_width=1.5,
        )
        tr_rect.move_to(axes2.c2p(theta_val, 0.5))
        tr_text = Text("Trust Region", font_size=16, color="#d2a8ff", font=_FONT)
        tr_text.next_to(tr_rect, UP, buff=0.15)

        self.play(FadeIn(tr_rect), FadeIn(tr_text))
        self.wait(0.5)

        # Safe step within trust region
        safe_val = -0.7
        safe_y = 1.5 * np.exp(-0.5 * safe_val**2) - 0.3
        safe_dot = Dot(axes2.c2p(safe_val, safe_y), color="#3fb950", radius=0.1)
        safe_label = Text("theta_new (safe!)", font_size=16, color="#3fb950", font=_FONT)
        safe_label.next_to(safe_dot, UP, buff=0.2)

        safe_arrow = Arrow(dot2.get_center(), safe_dot.get_center(),
                            color="#3fb950", stroke_width=3, buff=0.12)
        safe_step = Text("Bounded step", font_size=16, color="#3fb950", font=_FONT)
        safe_step.next_to(safe_arrow, DOWN, buff=0.15)

        self.play(GrowArrow(safe_arrow), FadeIn(safe_step))
        self.play(FadeIn(safe_dot), FadeIn(safe_label))
        self.wait(0.5)

        # Second iteration
        dot3 = safe_dot.copy()
        tr_rect2 = tr_rect.copy().move_to(axes2.c2p(safe_val, 0.5))
        self.play(FadeOut(tr_rect), FadeOut(tr_text), FadeOut(safe_arrow), FadeOut(safe_step),
                  FadeOut(dot2), FadeOut(dot2_label), FadeOut(safe_label))

        dot3_label = Text("theta_old", font_size=16, color="#58a6ff", font=_FONT)
        dot3_label.next_to(dot3, UP, buff=0.2)
        tr_text2 = Text("Trust Region", font_size=16, color="#d2a8ff", font=_FONT)
        tr_text2.next_to(tr_rect2, UP, buff=0.15)
        dot3.set_color("#58a6ff")
        self.play(FadeIn(tr_rect2), FadeIn(tr_text2), FadeIn(dot3_label))

        final_val = -0.1
        final_y = 1.5 * np.exp(-0.5 * final_val**2) - 0.3
        final_dot = Dot(axes2.c2p(final_val, final_y), color="#3fb950", radius=0.1)
        final_label = Text("theta_new", font_size=16, color="#3fb950", font=_FONT)
        final_label.next_to(final_dot, UP, buff=0.2)

        safe_arrow2 = Arrow(dot3.get_center(), final_dot.get_center(),
                             color="#3fb950", stroke_width=3, buff=0.12)
        self.play(GrowArrow(safe_arrow2))
        self.play(FadeIn(final_dot), FadeIn(final_label))
        self.wait(0.5)

        converge_text = Text("Steady, monotonic improvement toward optimum",
                              font_size=20, weight=BOLD, color="#f0883e", font=_FONT)
        converge_text.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(converge_text))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # SUMMARY
        # ════════════════════════════════════════
        summary = Text("Summary", font_size=30, weight=BOLD, color="#58a6ff", font=_FONT)
        summary.to_edge(UP, buff=0.5)
        self.play(Write(summary), run_time=0.5)

        items = VGroup(
            Text("Vanilla PG: no step size control -> can crash", font_size=20, color="#E74C3C", font=_FONT),
            Text("Trust Region: constrain KL divergence -> safe updates", font_size=20, color="#3fb950", font=_FONT),
            Text("TRPO: exact constraint (complex)", font_size=20, color="#d2a8ff", font=_FONT),
            Text("PPO: approximate via clipping (simple)", font_size=20, color="#f0883e", font=_FONT),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        items.next_to(summary, DOWN, buff=0.6)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.4)
            self.wait(0.3)

        box = SurroundingRectangle(items[3], color="#f0883e", buff=0.15, corner_radius=0.1)
        self.play(Create(box))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
