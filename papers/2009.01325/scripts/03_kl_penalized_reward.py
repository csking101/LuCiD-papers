# 03 — KL-Penalized Reward
#
# Tool: Manim
# Output: MP4
#
# Animated breakdown of the KL-penalized reward:
# R(x,y) = r_θ(x,y) - β·log[π_RL(y|x) / π_SFT(y|x)]
# Shows the two purposes: entropy bonus + staying near training distribution.
#
# Run:
#   manim -qm --media_dir ../output/animations 03_kl_penalized_reward.py KLPenalizedReward

from manim import *
import atexit
import shutil
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "2009.01325"

_FONT = "Latin Modern Roman"
_SUB = "#c9d1d9"  # high-contrast secondary text


def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/03_kl_penalized_reward/720p30/KLPenalizedReward.mp4"
    dst = _DOCS_DIR / "KLPenalizedReward.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


class KLPenalizedReward(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("The KL-Penalized Reward", font_size=36,
                      weight=BOLD, color=WHITE, font=_FONT)
        sub = Text("Balancing reward optimization with policy stability",
                    font_size=22, color=_SUB, font=_FONT)
        header = VGroup(title, sub).arrange(DOWN, buff=0.25).to_edge(UP, buff=0.5)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2), run_time=1.0)
        self.wait(0.6)

        # ══════════════════════════════════════════════════════════════
        # The full equation
        # ══════════════════════════════════════════════════════════════
        full_eq = MathTex(
            r"R(x,y)", r"=", r"r_\theta(x,y)", r"-",
            r"\beta", r"\log", r"\frac{\pi_\phi^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)}",
            font_size=34
        )
        full_eq.next_to(header, DOWN, buff=0.7)
        # Color each piece
        full_eq[0].set_color("#3fb950")  # R(x,y)
        full_eq[2].set_color("#f0883e")  # r_theta
        full_eq[4].set_color("#d2a8ff")  # beta
        full_eq[5].set_color("#d2a8ff")  # log
        full_eq[6].set_color("#d2a8ff")  # fraction

        self.play(Write(full_eq), run_time=1.5)
        self.wait(1.0)

        # ══════════════════════════════════════════════════════════════
        # Highlight Term 1: Reward Model Score
        # ══════════════════════════════════════════════════════════════
        box1 = SurroundingRectangle(full_eq[2], color="#f0883e", buff=0.15,
                                     stroke_width=2, corner_radius=0.05)
        label1 = Text("Reward Model Score", font_size=22, color="#f0883e",
                       weight=BOLD, font=_FONT)
        label1.next_to(box1, DOWN, buff=0.35)

        desc1 = VGroup(
            Text("• Learned from 65K human comparisons", font_size=20,
                 color=_SUB, font=_FONT),
            Text("• Higher = more preferred by humans", font_size=20,
                 color=_SUB, font=_FONT),
            Text("• The \"carrot\" — what we want to maximize", font_size=20,
                 color="#f0883e", font=_FONT),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        desc1.next_to(label1, DOWN, buff=0.35)

        self.play(Create(box1), FadeIn(label1), run_time=0.6)
        for d in desc1:
            self.play(FadeIn(d, shift=RIGHT * 0.2), run_time=0.4)
        self.wait(1.0)
        self.play(FadeOut(VGroup(box1, label1, desc1)), run_time=0.4)

        # ══════════════════════════════════════════════════════════════
        # Highlight Term 2: KL Penalty
        # ══════════════════════════════════════════════════════════════
        kl_group = VGroup(full_eq[3], full_eq[4], full_eq[5], full_eq[6])
        box2 = SurroundingRectangle(kl_group, color="#d2a8ff", buff=0.15,
                                     stroke_width=2, corner_radius=0.05)
        label2 = Text("KL Divergence Penalty", font_size=22, color="#d2a8ff",
                       weight=BOLD, font=_FONT)
        label2.next_to(box2, DOWN, buff=0.35)

        kl_expanded = MathTex(
            r"\text{KL}\!\left[\pi^{\text{RL}} \| \pi^{\text{SFT}}\right]"
            r" = \mathbb{E}\!\left[\log\frac{\pi^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)}\right]",
            font_size=26, color="#d2a8ff"
        )
        kl_expanded.next_to(label2, DOWN, buff=0.35)

        self.play(Create(box2), FadeIn(label2), run_time=0.6)
        self.play(Write(kl_expanded), run_time=0.8)
        self.wait(0.8)
        self.play(FadeOut(VGroup(box2, label2, kl_expanded)), run_time=0.4)

        # ══════════════════════════════════════════════════════════════
        # Purpose 1: Entropy bonus
        # ══════════════════════════════════════════════════════════════
        p1_title = Text("Purpose 1: Entropy Bonus", font_size=26,
                         color="#58a6ff", weight=BOLD, font=_FONT)
        p1_title.next_to(full_eq, DOWN, buff=0.6)

        p1_desc = VGroup(
            Text("Encourages exploration — prevents mode collapse",
                 font_size=20, color=_SUB, font=_FONT),
            Text("Without it: policy collapses to a single \"best\" output",
                 font_size=20, color="#E74C3C", font=_FONT),
            Text("With it: policy maintains diversity in generations",
                 font_size=20, color="#3fb950", font=_FONT),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        p1_desc.next_to(p1_title, DOWN, buff=0.35)

        self.play(FadeIn(p1_title), run_time=0.5)
        for d in p1_desc:
            self.play(FadeIn(d, shift=RIGHT * 0.2), run_time=0.4)
        self.wait(1.0)
        self.play(FadeOut(VGroup(p1_title, p1_desc)), run_time=0.4)

        # ══════════════════════════════════════════════════════════════
        # Purpose 2: Stay near training distribution
        # ══════════════════════════════════════════════════════════════
        p2_title = Text("Purpose 2: Stay Near SFT Distribution", font_size=26,
                         color="#58a6ff", weight=BOLD, font=_FONT)
        p2_title.next_to(full_eq, DOWN, buff=0.6)

        p2_desc = VGroup(
            Text("Reward model only trained on outputs near π_SFT",
                 font_size=20, color=_SUB, font=_FONT),
            Text("Drifting too far → reward model gives garbage scores",
                 font_size=20, color="#E74C3C", font=_FONT),
            Text("KL penalty keeps RL policy in the \"trusted\" region",
                 font_size=20, color="#3fb950", font=_FONT),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        p2_desc.next_to(p2_title, DOWN, buff=0.35)

        self.play(FadeIn(p2_title), run_time=0.5)
        for d in p2_desc:
            self.play(FadeIn(d, shift=RIGHT * 0.2), run_time=0.4)
        self.wait(1.0)
        self.play(FadeOut(VGroup(p2_title, p2_desc)), run_time=0.4)

        # ══════════════════════════════════════════════════════════════
        # The β tradeoff
        # ══════════════════════════════════════════════════════════════
        beta_title = Text("The β Tradeoff", font_size=26,
                           color="#d2a8ff", weight=BOLD, font=_FONT)
        beta_title.next_to(full_eq, DOWN, buff=0.6)

        tradeoff = VGroup(
            VGroup(
                MathTex(r"\beta \to 0", font_size=26, color="#E74C3C"),
                Text(": maximize reward aggressively (may overoptimize)",
                     font_size=20, color="#E74C3C", font=_FONT),
            ).arrange(RIGHT, buff=0.15),
            VGroup(
                MathTex(r"\beta \to \infty", font_size=26, color="#58a6ff"),
                Text(": stay very close to SFT (safe but no improvement)",
                     font_size=20, color="#58a6ff", font=_FONT),
            ).arrange(RIGHT, buff=0.15),
            VGroup(
                MathTex(r"\beta^*", font_size=26, color="#3fb950"),
                Text(": sweet spot — improve quality without gaming the RM",
                     font_size=20, color="#3fb950", font=_FONT),
            ).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        tradeoff.next_to(beta_title, DOWN, buff=0.4)

        self.play(FadeIn(beta_title), run_time=0.5)
        for t in tradeoff:
            self.play(FadeIn(t, shift=RIGHT * 0.2), run_time=0.5)
        self.wait(2.0)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
