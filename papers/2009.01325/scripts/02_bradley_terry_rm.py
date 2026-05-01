# 02 — Bradley-Terry Reward Model
#
# Tool: Manim
# Output: MP4
#
# Animated derivation of the reward model loss:
# Two summaries → scalar scores → sigmoid on difference → probability → loss.
# Connects to the Bradley-Terry preference model from the RLHF paper.
#
# Run:
#   manim -ql --media_dir ../output/animations 02_bradley_terry_rm.py BradleyTerryRM

from manim import *
import atexit
import shutil
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "2009.01325"

_FONT = "Latin Modern Roman"


def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/02_bradley_terry_rm/480p15/BradleyTerryRM.mp4"
    dst = _DOCS_DIR / "BradleyTerryRM.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


class BradleyTerryRM(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("Bradley-Terry Reward Model", font_size=34,
                      weight=BOLD, color=WHITE, font=_FONT)
        sub = Text("How human preferences become a trainable loss",
                    font_size=18, color="#8b949e", font=_FONT)
        header = VGroup(title, sub).arrange(DOWN, buff=0.2).to_edge(UP, buff=0.5)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2), run_time=1.0)
        self.wait(0.6)

        # ══════════════════════════════════════════════════════════════
        # Step 1: Setup — two summaries
        # ══════════════════════════════════════════════════════════════
        step1 = Text("Step 1: Two candidate summaries", font_size=22,
                      color="#58a6ff", font=_FONT)
        step1.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(step1))

        post_box = RoundedRectangle(corner_radius=0.1, width=2.5, height=0.8,
                                     stroke_color="#8b949e", fill_color="#161b22", fill_opacity=0.8)
        post_label = Text("Post x", font_size=18, color="#c9d1d9", font=_FONT)
        post_label.move_to(post_box)
        post = VGroup(post_box, post_label)

        y0_box = RoundedRectangle(corner_radius=0.1, width=2.2, height=0.8,
                                   stroke_color="#E74C3C", fill_color="#E74C3C", fill_opacity=0.1)
        y0_label = Text("Summary y₀", font_size=16, color="#E74C3C", font=_FONT)
        y0_label.move_to(y0_box)
        y0 = VGroup(y0_box, y0_label)

        y1_box = RoundedRectangle(corner_radius=0.1, width=2.2, height=0.8,
                                   stroke_color="#3fb950", fill_color="#3fb950", fill_opacity=0.1)
        y1_label = Text("Summary y₁ ✓", font_size=16, color="#3fb950", font=_FONT)
        y1_label.move_to(y1_box)
        y1 = VGroup(y1_box, y1_label)

        inputs = VGroup(post, y0, y1).arrange(RIGHT, buff=0.6)
        inputs.next_to(step1, DOWN, buff=0.4)

        preferred = Text("(human prefers y₁)", font_size=14, color="#3fb950", font=_FONT)
        preferred.next_to(inputs, DOWN, buff=0.2)

        self.play(FadeIn(post), FadeIn(y0), FadeIn(y1), run_time=0.8)
        self.play(FadeIn(preferred), run_time=0.4)
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════
        # Step 2: Reward model scores each summary
        # ══════════════════════════════════════════════════════════════
        self.play(FadeOut(VGroup(step1, inputs, preferred)), run_time=0.4)

        step2 = Text("Step 2: Reward model scores each summary", font_size=22,
                      color="#f0883e", font=_FONT)
        step2.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(step2))

        # Show r(x, y0) and r(x, y1)
        score0 = MathTex(r"r_\theta(x, y_0) = 1.2", font_size=30, color="#E74C3C")
        score1 = MathTex(r"r_\theta(x, y_1) = 3.8", font_size=30, color="#3fb950")
        scores = VGroup(score0, score1).arrange(RIGHT, buff=1.5)
        scores.next_to(step2, DOWN, buff=0.6)

        diff_eq = MathTex(
            r"\Delta = r_\theta(x, y_1) - r_\theta(x, y_0) = 2.6",
            font_size=28, color="#f0883e"
        )
        diff_eq.next_to(scores, DOWN, buff=0.5)

        self.play(Write(score0), Write(score1), run_time=0.8)
        self.wait(0.4)
        self.play(Write(diff_eq), run_time=0.8)
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════
        # Step 3: Sigmoid converts difference to probability
        # ══════════════════════════════════════════════════════════════
        self.play(FadeOut(VGroup(step2, scores, diff_eq)), run_time=0.4)

        step3 = Text("Step 3: Sigmoid → probability", font_size=22,
                      color="#d2a8ff", font=_FONT)
        step3.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(step3))

        prob_eq = MathTex(
            r"P(y_1 \succ y_0) = \sigma(\Delta) = \sigma(2.6) \approx 0.93",
            font_size=28, color="#d2a8ff"
        )
        prob_eq.next_to(step3, DOWN, buff=0.6)

        interp = Text(
            "\"93% chance the preferred summary scores higher\"",
            font_size=16, color="#8b949e", font=_FONT
        )
        interp.next_to(prob_eq, DOWN, buff=0.4)

        general_eq = MathTex(
            r"\hat{P}[\sigma^1 \succ \sigma^2] = "
            r"\frac{e^{r_\theta(x,y_1)}}{e^{r_\theta(x,y_0)} + e^{r_\theta(x,y_1)}}",
            font_size=26, color=WHITE
        )
        general_eq.next_to(interp, DOWN, buff=0.5)

        self.play(Write(prob_eq), run_time=0.8)
        self.play(FadeIn(interp), run_time=0.5)
        self.play(Write(general_eq), run_time=1.0)
        self.wait(1.0)

        # ══════════════════════════════════════════════════════════════
        # Step 4: Cross-entropy loss
        # ══════════════════════════════════════════════════════════════
        self.play(FadeOut(VGroup(step3, prob_eq, interp, general_eq)), run_time=0.4)

        step4 = Text("Step 4: Cross-entropy loss", font_size=22,
                      color="#3fb950", font=_FONT)
        step4.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(step4))

        loss = MathTex(
            r"\text{loss}(r_\theta) = "
            r"-\mathbb{E}_{(x,y_0,y_1,i)\sim D}"
            r"\left[\log\sigma\!\left("
            r"r_\theta(x,y_i) - r_\theta(x,y_{1-i})"
            r"\right)\right]",
            font_size=24, color="#3fb950"
        )
        loss.next_to(step4, DOWN, buff=0.6)

        bullets = VGroup(
            Text("• Maximize log-prob that preferred summary scores higher",
                 font_size=16, color="#c9d1d9", font=_FONT),
            Text("• Only the difference matters — absolute scale is arbitrary",
                 font_size=16, color="#c9d1d9", font=_FONT),
            Text("• Same as Bradley-Terry / Elo rating systems",
                 font_size=16, color="#c9d1d9", font=_FONT),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        bullets.next_to(loss, DOWN, buff=0.5)

        self.play(Write(loss), run_time=1.2)
        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.4)
        self.wait(1.5)

        # ── Numeric example ──
        self.play(FadeOut(VGroup(step4, loss, bullets)), run_time=0.4)

        ex_title = Text("Putting it together", font_size=22,
                         color=WHITE, weight=BOLD, font=_FONT)
        ex_title.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(ex_title))

        lines = VGroup(
            MathTex(r"r_\theta(x, y_1) - r_\theta(x, y_0) = 3.8 - 1.2 = 2.6",
                    font_size=24, color="#f0883e"),
            MathTex(r"\sigma(2.6) \approx 0.93",
                    font_size=24, color="#d2a8ff"),
            MathTex(r"\text{loss} = -\log(0.93) \approx 0.07 \quad \text{(low — correct!)}",
                    font_size=24, color="#3fb950"),
        ).arrange(DOWN, buff=0.4)
        lines.next_to(ex_title, DOWN, buff=0.5)

        for line in lines:
            self.play(Write(line), run_time=0.7)
            self.wait(0.3)

        wrong = MathTex(
            r"\text{If wrong: } \sigma(-2.6) \approx 0.07 \;\Rightarrow\; "
            r"\text{loss} = -\log(0.07) \approx 2.66 \quad \text{(high!)}",
            font_size=20, color="#E74C3C"
        )
        wrong.next_to(lines, DOWN, buff=0.5)
        self.play(Write(wrong), run_time=0.8)
        self.wait(2.0)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
