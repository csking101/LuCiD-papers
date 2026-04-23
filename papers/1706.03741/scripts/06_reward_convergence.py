"""
Visualization 6: Reward Learning Convergence
Manim animation showing how the learned reward function approximates
the true reward as more human comparisons are collected.

Shows:
- True reward function (hidden from agent)
- Learned reward with 10, 50, 200, 700 comparisons
- Confidence bands narrowing with more data
- Key insight: reward shape matters more than exact values

Run: manim -ql --media_dir ../output/animations 06_reward_convergence.py RewardConvergence
"""

from manim import *
import numpy as np
import atexit
import shutil
from pathlib import Path

# Post-render: copy MP4 to docs/ for GitHub Pages serving
_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "1706.03741"

def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/06_reward_convergence/480p15/RewardConvergence.mp4"
    dst = _DOCS_DIR / "RewardConvergence.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst}")

atexit.register(_copy_to_docs)


class RewardConvergence(Scene):
    def construct(self):
        # Title
        title = Text("Reward Learning Convergence", font_size=38, weight=BOLD)
        subtitle = Text(
            "How the reward model improves with more human comparisons",
            font_size=20, color=GREY
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1.5, 2, 0.5],
            x_length=10,
            y_length=5,
            axis_config={"color": GREY, "include_numbers": False, "font_size": 20},
        ).move_to(DOWN * 0.3)

        x_label = Text("State/Action Features", font_size=16, color=GREY)
        x_label.next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Text("Reward", font_size=16, color=GREY)
        y_label.next_to(axes.y_axis, LEFT, buff=0.3).rotate(PI / 2)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))

        # True reward function (a smooth curve the agent doesn't see)
        def true_reward(x):
            return 0.5 * np.sin(1.5 * x) + 0.3 * x + 0.2

        true_graph = axes.plot(
            true_reward, x_range=[-3, 3], color=GREEN, stroke_width=3
        )
        true_label = Text("True Reward r(s,a)", font_size=16, color=GREEN, weight=BOLD)
        true_label.move_to(UP * 3)

        self.play(Create(true_graph), FadeIn(true_label), run_time=1)
        self.wait(0.5)

        # === Progressive approximations ===
        stages = [
            (10, RED, 0.8, "10 comparisons — very noisy"),
            (50, ORANGE, 0.5, "50 comparisons — capturing shape"),
            (200, YELLOW, 0.3, "200 comparisons — good approximation"),
            (700, BLUE, 0.1, "700 comparisons — nearly matches!"),
        ]

        prev_graph = None
        prev_band = None
        prev_label = None

        for n_comp, color, noise_level, desc_text in stages:
            # Counter
            counter = Text(f"n = {n_comp}", font_size=26, weight=BOLD, color=color)
            counter.move_to(RIGHT * 4 + UP * 2.5)

            desc = Text(desc_text, font_size=18, color=color)
            desc.move_to(RIGHT * 3 + UP * 2)

            # Generate noisy approximation
            rng = np.random.RandomState(n_comp)

            def learned_reward(x, _noise=noise_level, _rng=rng, _n=n_comp):
                base = true_reward(x)
                # Add systematic bias that decreases with more data
                bias = _noise * 0.5 * np.sin(3 * x + _n * 0.1)
                return base + bias

            learned_graph = axes.plot(
                learned_reward, x_range=[-3, 3], color=color, stroke_width=2.5
            )

            # Confidence band
            upper = axes.plot(
                lambda x, _nl=noise_level: learned_reward(x) + _nl * 0.8,
                x_range=[-3, 3], color=color, stroke_width=0.5, stroke_opacity=0.3
            )
            lower = axes.plot(
                lambda x, _nl=noise_level: learned_reward(x) - _nl * 0.8,
                x_range=[-3, 3], color=color, stroke_width=0.5, stroke_opacity=0.3
            )
            band = axes.get_area(upper, bounded_graph=lower, color=color, opacity=0.1)

            # Comparison dots (showing where comparisons were made)
            x_samples = rng.uniform(-2.5, 2.5, min(n_comp // 2, 20))
            dots = VGroup(*[
                Dot(axes.c2p(x, true_reward(x)), radius=0.04, color=color, fill_opacity=0.5)
                for x in x_samples
            ])

            # Animate
            anims_out = []
            if prev_graph:
                anims_out.extend([FadeOut(prev_graph), FadeOut(prev_band), FadeOut(prev_label)])

            self.play(
                *anims_out,
                Create(learned_graph),
                FadeIn(band),
                FadeIn(dots),
                FadeIn(counter),
                FadeIn(desc),
                run_time=1.2
            )
            self.wait(1)

            # Clean up for next iteration
            self.play(FadeOut(dots), FadeOut(counter), FadeOut(desc), run_time=0.3)
            prev_graph = learned_graph
            prev_band = band
            prev_label = None

        # Final comparison
        final_text = VGroup(
            Text("Key Insight:", font_size=22, weight=BOLD, color=WHITE),
            Text("The reward model doesn't need to be perfect —", font_size=18, color=GREY),
            Text("it just needs the right shape for policy optimization.", font_size=18, color=GREY),
            Text("", font_size=10),
            Text("With 700 comparisons, the learned reward nearly", font_size=18, color=BLUE),
            Text("matches the true reward across the state space.", font_size=18, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        final_text.move_to(RIGHT * 2.5 + DOWN * 2.5).scale(0.8)

        self.play(FadeIn(final_text, shift=UP * 0.3))
        self.wait(3)
