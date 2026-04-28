# 02 — Policy Gradient Intuition
#
# Tool: Manim
# Output: MP4
#
# Animated walkthrough of the REINFORCE loop:
# Agent in a state → policy outputs action probabilities → action taken
# → reward received → gradient pushes good actions up, bad actions down.
#
# Run:
#   manim -ql --media_dir ../output/animations 02_policy_gradient_intuition.py PolicyGradientIntuition

from manim import *
import atexit
import shutil
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "1707.06347"

# Font for descriptive text — matches LaTeX aesthetic
_FONT = "Latin Modern Roman"


def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/02_policy_gradient_intuition/480p15/PolicyGradientIntuition.mp4"
    dst = _DOCS_DIR / "PolicyGradientIntuition.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


class PolicyGradientIntuition(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("Policy Gradient: The Intuition", font_size=36, weight=BOLD,
                      color=WHITE, font=_FONT)
        self.play(Write(title), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(title))

        # ════════════════════════════════════════
        # STEP 1: The Setup
        # ════════════════════════════════════════
        step1 = Text("Step 1: The Agent and Environment", font_size=28, weight=BOLD,
                      color="#58a6ff", font=_FONT)
        step1.to_edge(UP, buff=0.4)
        self.play(Write(step1), run_time=0.6)

        # Agent box
        agent_box = RoundedRectangle(width=2.5, height=1.2, corner_radius=0.15,
                                      color="#58a6ff", fill_opacity=0.15)
        agent_label = Text("Agent", font_size=20, weight=BOLD, color="#58a6ff", font=_FONT)
        agent_label.move_to(agent_box)
        agent = VGroup(agent_box, agent_label).move_to(LEFT * 3)

        # Policy inside agent — LaTeX
        policy_label = MathTex(r"\pi_\theta(a \mid s)", font_size=26, color=GREY)
        policy_label.next_to(agent_box, DOWN, buff=0.15)

        # Environment box
        env_box = RoundedRectangle(width=2.5, height=1.2, corner_radius=0.15,
                                    color="#3fb950", fill_opacity=0.15)
        env_label = Text("Environment", font_size=20, weight=BOLD, color="#3fb950", font=_FONT)
        env_label.move_to(env_box)
        env = VGroup(env_box, env_label).move_to(RIGHT * 3)

        self.play(FadeIn(agent), FadeIn(env), FadeIn(policy_label))
        self.wait(0.5)

        # Arrows: state and action
        state_arrow = Arrow(env_box.get_left() + UP * 0.3, agent_box.get_right() + UP * 0.3,
                            color="#3fb950", buff=0.1, stroke_width=3)
        state_text = MathTex(r"s_t", font_size=28, color="#3fb950")
        state_text.next_to(state_arrow, UP, buff=0.1)

        action_arrow = Arrow(agent_box.get_right() + DOWN * 0.3, env_box.get_left() + DOWN * 0.3,
                              color="#58a6ff", buff=0.1, stroke_width=3)
        action_text = MathTex(r"a_t", font_size=28, color="#58a6ff")
        action_text.next_to(action_arrow, DOWN, buff=0.1)

        self.play(GrowArrow(state_arrow), FadeIn(state_text))
        self.wait(0.3)
        self.play(GrowArrow(action_arrow), FadeIn(action_text))

        # Reward
        reward_text = MathTex(r"r_t", font_size=28, color="#f0883e")
        reward_text.next_to(state_text, UP, buff=0.3)
        reward_arrow = Arrow(env_box.get_top(), agent_box.get_top(),
                              color="#f0883e", buff=0.1, stroke_width=3,
                              path_arc=-0.5)
        self.play(GrowArrow(reward_arrow), FadeIn(reward_text))
        self.wait(1)

        self.play(*[FadeOut(m) for m in self.mobjects if m != step1])
        self.play(FadeOut(step1))

        # ════════════════════════════════════════
        # STEP 2: Policy outputs probabilities
        # ════════════════════════════════════════
        step2 = Text("Step 2: Policy Outputs Action Probabilities", font_size=28,
                      weight=BOLD, color="#58a6ff", font=_FONT)
        step2.to_edge(UP, buff=0.4)
        self.play(Write(step2), run_time=0.6)

        state_box = RoundedRectangle(width=2, height=0.8, corner_radius=0.1,
                                      color="#3fb950", fill_opacity=0.2)
        state_lab = MathTex(r"s_t", font_size=32, color="#3fb950")
        state_lab.move_to(state_box)
        state_g = VGroup(state_box, state_lab).move_to(LEFT * 4.5)

        # Neural net
        nn_box = RoundedRectangle(width=2, height=1.5, corner_radius=0.15,
                                   color="#d2a8ff", fill_opacity=0.15)
        nn_label = Text("Neural Net", font_size=16, weight=BOLD, color="#d2a8ff", font=_FONT)
        nn_theta = MathTex(r"(\theta)", font_size=26, color=GREY)
        VGroup(nn_label, nn_theta).arrange(DOWN, buff=0.1).move_to(nn_box)
        nn = VGroup(nn_box, nn_label, nn_theta).move_to(LEFT * 1)

        self.play(FadeIn(state_g), FadeIn(nn))

        arr1 = Arrow(state_box.get_right(), nn_box.get_left(), buff=0.1, color=GREY, stroke_width=2)
        self.play(GrowArrow(arr1))

        # Output: bar chart of action probs
        actions = ["Left", "Right", "Up"]
        probs = [0.2, 0.6, 0.2]
        colors_act = ["#58a6ff", "#3fb950", "#f0883e"]

        bars = VGroup()
        bar_labels = VGroup()
        prob_labels = VGroup()
        for i, (act, p, c) in enumerate(zip(actions, probs, colors_act)):
            bar = Rectangle(width=0.6, height=p * 3, color=c, fill_opacity=0.7)
            bar.move_to(RIGHT * (2.5 + i * 1.0) + DOWN * (1.5 - p * 1.5))
            bars.add(bar)

            lab = Text(act, font_size=14, color=c, font=_FONT)
            lab.next_to(bar, DOWN, buff=0.1)
            bar_labels.add(lab)

            plab = Text(f"{p:.0%}", font_size=14, color=WHITE, font=_FONT)
            plab.next_to(bar, UP, buff=0.1)
            prob_labels.add(plab)

        arr2 = Arrow(nn_box.get_right(), RIGHT * 1.8 + DOWN * 0.3, buff=0.1, color=GREY, stroke_width=2)
        self.play(GrowArrow(arr2))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.15))
        self.play(FadeIn(bar_labels), FadeIn(prob_labels))
        self.wait(0.5)

        # Highlight "Right" as sampled
        highlight = SurroundingRectangle(bars[1], color="#3fb950", buff=0.08)
        sampled = Text("Sampled!", font_size=16, weight=BOLD, color="#3fb950", font=_FONT)
        sampled.next_to(highlight, UP, buff=0.3)
        self.play(Create(highlight), FadeIn(sampled))
        self.wait(1)

        self.play(*[FadeOut(m) for m in self.mobjects if m != step2])
        self.play(FadeOut(step2))

        # ════════════════════════════════════════
        # STEP 3: Reward signal adjusts probabilities
        # ════════════════════════════════════════
        step3 = Text("Step 3: Reward Adjusts Future Probabilities", font_size=28,
                      weight=BOLD, color="#58a6ff", font=_FONT)
        step3.to_edge(UP, buff=0.4)
        self.play(Write(step3), run_time=0.6)

        # Good reward scenario — LaTeX for A > 0
        good_label_parts = VGroup(
            Text("Good reward ", font_size=20, color="#3fb950", font=_FONT),
            MathTex(r"(\hat{A} > 0)", font_size=28, color="#3fb950"),
            Text(": increase P(Right)", font_size=20, color="#3fb950", font=_FONT),
        ).arrange(RIGHT, buff=0.1)
        good_label_parts.next_to(step3, DOWN, buff=0.6)
        self.play(FadeIn(good_label_parts))

        # Before bars
        before_label = Text("Before", font_size=16, color=GREY, font=_FONT)
        before_label.move_to(LEFT * 3.5 + UP * 0.3)

        probs_before = [0.2, 0.6, 0.2]
        bars_before = VGroup()
        for i, (p, c) in enumerate(zip(probs_before, colors_act)):
            bar = Rectangle(width=0.5, height=p * 2.5, color=c, fill_opacity=0.5)
            bar.move_to(LEFT * (4.5 - i * 0.8) + DOWN * (0.8 - p * 1.25))
            bars_before.add(bar)
            lab = Text(f"{p:.0%}", font_size=12, color=WHITE, font=_FONT)
            lab.next_to(bar, UP, buff=0.05)
            bars_before.add(lab)

        # After bars (good reward)
        after_label = Text("After", font_size=16, color=GREY, font=_FONT)
        after_label.move_to(RIGHT * 2.5 + UP * 0.3)

        probs_after = [0.15, 0.72, 0.13]
        bars_after = VGroup()
        for i, (p, c) in enumerate(zip(probs_after, colors_act)):
            bar = Rectangle(width=0.5, height=p * 2.5, color=c, fill_opacity=0.8)
            bar.move_to(RIGHT * (1.5 + i * 0.8) + DOWN * (0.8 - p * 1.25))
            bars_after.add(bar)
            lab = Text(f"{p:.0%}", font_size=12, color=WHITE, font=_FONT)
            lab.next_to(bar, UP, buff=0.05)
            bars_after.add(lab)

        update_arrow = Arrow(LEFT * 2.5 + DOWN * 0.5, RIGHT * 0.5 + DOWN * 0.5,
                              color="#3fb950", buff=0.1, stroke_width=3)
        update_text = Text("gradient ascent", font_size=14, color="#3fb950", font=_FONT)
        update_text.next_to(update_arrow, DOWN, buff=0.1)

        self.play(FadeIn(before_label), FadeIn(bars_before))
        self.play(GrowArrow(update_arrow), FadeIn(update_text))
        self.play(FadeIn(after_label), FadeIn(bars_after))
        self.wait(1)

        # Bad reward scenario
        self.play(*[FadeOut(m) for m in self.mobjects if m not in [step3]])

        bad_label_parts = VGroup(
            Text("Bad reward ", font_size=20, color="#E74C3C", font=_FONT),
            MathTex(r"(\hat{A} < 0)", font_size=28, color="#E74C3C"),
            Text(": decrease P(Right)", font_size=20, color="#E74C3C", font=_FONT),
        ).arrange(RIGHT, buff=0.1)
        bad_label_parts.next_to(step3, DOWN, buff=0.6)
        self.play(FadeIn(bad_label_parts))

        probs_bad = [0.3, 0.4, 0.3]
        bars_bad = VGroup()
        for i, (p, c) in enumerate(zip(probs_bad, colors_act)):
            bar = Rectangle(width=0.5, height=p * 2.5, color=c, fill_opacity=0.8)
            bar.move_to(RIGHT * (1.5 + i * 0.8) + DOWN * (0.8 - p * 1.25))
            bars_bad.add(bar)
            lab = Text(f"{p:.0%}", font_size=12, color=WHITE, font=_FONT)
            lab.next_to(bar, UP, buff=0.05)
            bars_bad.add(lab)

        before_label2 = Text("Before", font_size=16, color=GREY, font=_FONT)
        before_label2.move_to(LEFT * 3.5 + UP * 0.3)

        bars_before2 = VGroup()
        for i, (p, c) in enumerate(zip(probs_before, colors_act)):
            bar = Rectangle(width=0.5, height=p * 2.5, color=c, fill_opacity=0.5)
            bar.move_to(LEFT * (4.5 - i * 0.8) + DOWN * (0.8 - p * 1.25))
            bars_before2.add(bar)
            lab = Text(f"{p:.0%}", font_size=12, color=WHITE, font=_FONT)
            lab.next_to(bar, UP, buff=0.05)
            bars_before2.add(lab)

        after_label2 = Text("After", font_size=16, color=GREY, font=_FONT)
        after_label2.move_to(RIGHT * 2.5 + UP * 0.3)

        update_arrow2 = Arrow(LEFT * 2.5 + DOWN * 0.5, RIGHT * 0.5 + DOWN * 0.5,
                               color="#E74C3C", buff=0.1, stroke_width=3)
        update_text2 = Text("gradient ascent", font_size=14, color="#E74C3C", font=_FONT)
        update_text2.next_to(update_arrow2, DOWN, buff=0.1)

        self.play(FadeIn(before_label2), FadeIn(bars_before2))
        self.play(GrowArrow(update_arrow2), FadeIn(update_text2))
        self.play(FadeIn(after_label2), FadeIn(bars_bad))
        self.wait(1)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # STEP 4: The REINFORCE Loop
        # ════════════════════════════════════════
        step4 = Text("The REINFORCE Loop", font_size=28, weight=BOLD, color="#58a6ff", font=_FONT)
        step4.to_edge(UP, buff=0.4)
        self.play(Write(step4), run_time=0.6)

        steps = [
            ("1", "Run policy, collect trajectories", "#58a6ff"),
            ("2", "Compute rewards for each (s, a) pair", "#3fb950"),
        ]

        step_group = VGroup()
        for num, text, color in steps:
            row = VGroup(
                Text(num + ".", font_size=20, weight=BOLD, color=color, font=_FONT),
                Text(text, font_size=18, color=WHITE, font=_FONT),
            ).arrange(RIGHT, buff=0.2)
            step_group.add(row)

        # Step 3 uses LaTeX for the gradient expression
        row3_num = Text("3.", font_size=20, weight=BOLD, color="#d2a8ff", font=_FONT)
        row3_text = Text("Compute gradient: ", font_size=18, color=WHITE, font=_FONT)
        row3_math = MathTex(r"R \cdot \nabla_\theta \log \pi_\theta(a|s)",
                            font_size=28, color="#d2a8ff")
        row3 = VGroup(row3_num, row3_text, row3_math).arrange(RIGHT, buff=0.15)
        step_group.add(row3)

        more_steps = [
            ("4", "Update theta in direction of gradient", "#f0883e"),
            ("5", "Repeat", GREY),
        ]
        for num, text, color in more_steps:
            row = VGroup(
                Text(num + ".", font_size=20, weight=BOLD, color=color, font=_FONT),
                Text(text, font_size=18, color=WHITE, font=_FONT),
            ).arrange(RIGHT, buff=0.2)
            step_group.add(row)

        step_group.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        step_group.next_to(step4, DOWN, buff=0.6)

        for row in step_group:
            self.play(FadeIn(row, shift=RIGHT * 0.3), run_time=0.4)
            self.wait(0.2)

        key_insight = Text("Key: good actions get reinforced, bad actions get suppressed",
                           font_size=18, weight=BOLD, color="#f0883e", font=_FONT)
        key_insight.to_edge(DOWN, buff=0.6)
        box = SurroundingRectangle(key_insight, color="#f0883e", buff=0.15, corner_radius=0.1)
        self.play(FadeIn(key_insight), Create(box))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])
