# 01 — RLHF-for-Text Pipeline
#
# Tool: Manim
# Output: MP4
#
# Animated walkthrough of the 3-stage pipeline from this paper:
# Stage 1: Supervised Fine-Tuning (SFT) on Reddit TL;DR
# Stage 2: Reward Model training from 65K human comparisons
# Stage 3: PPO fine-tuning with KL penalty
#
# Run:
#   manim -qm --media_dir ../output/animations 01_rlhf_text_pipeline.py RLHFTextPipeline

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
    src = _PAPER_DIR / "output/animations/videos/01_rlhf_text_pipeline/720p30/RLHFTextPipeline.mp4"
    dst = _DOCS_DIR / "RLHFTextPipeline.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


def _box(label, color, width=2.8, height=1.2):
    """Create a rounded rectangle with centered label."""
    rect = RoundedRectangle(
        corner_radius=0.15, width=width, height=height,
        stroke_color=color, fill_color=color, fill_opacity=0.15,
        stroke_width=2
    )
    txt = Text(label, font_size=22, color=color, font=_FONT)
    txt.move_to(rect.get_center())
    return VGroup(rect, txt)


def _arrow_between(a, b, color=WHITE):
    return Arrow(
        a.get_right(), b.get_left(),
        buff=0.15, color=color, stroke_width=2, max_tip_length_to_length_ratio=0.15
    )


def _small_label(text, color=_SUB, size=18):
    return Text(text, font_size=size, color=color, font=_FONT)


class RLHFTextPipeline(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("RLHF for Text Summarization", font_size=36,
                      weight=BOLD, color=WHITE, font=_FONT)
        sub = Text("Stiennon et al., 2020 — 3-Stage Pipeline",
                    font_size=22, color=_SUB, font=_FONT)
        header = VGroup(title, sub).arrange(DOWN, buff=0.25).to_edge(UP, buff=0.5)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2), run_time=1.2)
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════
        # STAGE 1 — Supervised Fine-Tuning
        # ══════════════════════════════════════════════════════════════
        stage1_label = Text("Stage 1: Supervised Fine-Tuning (SFT)",
                            font_size=26, color="#58a6ff", weight=BOLD, font=_FONT)
        stage1_label.next_to(header, DOWN, buff=0.6)
        self.play(FadeIn(stage1_label, shift=RIGHT * 0.3))

        # Data → GPT-3 → SFT Policy
        data_box = _box("Reddit TL;DR\n(123K posts)", "#58a6ff", width=2.6, height=1.0)
        gpt_box = _box("GPT-3\n(pretrained)", "#d2a8ff", width=2.6, height=1.0)
        sft_box = _box("SFT Policy\nπ_SFT", "#3fb950", width=2.6, height=1.0)

        row1 = VGroup(data_box, gpt_box, sft_box).arrange(RIGHT, buff=1.2)
        row1.next_to(stage1_label, DOWN, buff=0.5)

        a1 = _arrow_between(data_box, gpt_box, "#58a6ff")
        a2 = _arrow_between(gpt_box, sft_box, "#d2a8ff")

        lbl1 = _small_label("fine-tune on\nhuman summaries")
        lbl1.next_to(a1, UP, buff=0.15)
        lbl2 = _small_label("next-token\nprediction loss")
        lbl2.next_to(a2, UP, buff=0.15)

        self.play(
            FadeIn(data_box, shift=UP * 0.3),
            FadeIn(gpt_box, shift=UP * 0.3),
            run_time=0.8
        )
        self.play(Create(a1), FadeIn(lbl1), run_time=0.6)
        self.play(Create(a2), FadeIn(lbl2), run_time=0.6)
        self.play(FadeIn(sft_box, shift=UP * 0.3), run_time=0.6)
        self.wait(1.0)

        # Fade out stage 1 content, keep SFT box reference
        stage1_all = VGroup(stage1_label, data_box, gpt_box, sft_box, a1, a2, lbl1, lbl2)
        self.play(FadeOut(stage1_all), run_time=0.6)

        # ══════════════════════════════════════════════════════════════
        # STAGE 2 — Reward Model
        # ══════════════════════════════════════════════════════════════
        stage2_label = Text("Stage 2: Reward Model Training",
                            font_size=26, color="#f0883e", weight=BOLD, font=_FONT)
        stage2_label.next_to(header, DOWN, buff=0.6)
        self.play(FadeIn(stage2_label, shift=RIGHT * 0.3))

        # Summaries → Human Comparisons → Reward Model
        sum_box = _box("Summary Pairs\n(y₀, y₁)", "#58a6ff", width=2.6, height=1.0)
        human_box = _box("Human Labelers\n65K comparisons", "#f0883e", width=2.8, height=1.0)
        rm_box = _box("Reward Model\nr_θ(x, y)", "#f0883e", width=2.6, height=1.0)

        row2 = VGroup(sum_box, human_box, rm_box).arrange(RIGHT, buff=1.0)
        row2.next_to(stage2_label, DOWN, buff=0.5)

        a3 = _arrow_between(sum_box, human_box, "#58a6ff")
        a4 = _arrow_between(human_box, rm_box, "#f0883e")

        lbl3 = _small_label("which is better?")
        lbl3.next_to(a3, UP, buff=0.15)
        lbl4 = _small_label("Bradley-Terry\ncross-entropy")
        lbl4.next_to(a4, UP, buff=0.15)

        self.play(FadeIn(sum_box, shift=UP * 0.3), run_time=0.6)
        self.play(Create(a3), FadeIn(lbl3), FadeIn(human_box, shift=UP * 0.3), run_time=0.8)
        self.play(Create(a4), FadeIn(lbl4), run_time=0.6)
        self.play(FadeIn(rm_box, shift=UP * 0.3), run_time=0.6)

        # Show the loss equation
        loss_eq = MathTex(
            r"\text{loss} = -\mathbb{E}\left[\log\sigma\!\left(r_\theta(x,y_i) - r_\theta(x,y_{1-i})\right)\right]",
            font_size=30, color="#f0883e"
        )
        loss_eq.next_to(row2, DOWN, buff=0.5)
        self.play(Write(loss_eq), run_time=1.0)
        self.wait(1.2)

        stage2_all = VGroup(stage2_label, sum_box, human_box, rm_box, a3, a4, lbl3, lbl4, loss_eq)
        self.play(FadeOut(stage2_all), run_time=0.6)

        # ══════════════════════════════════════════════════════════════
        # STAGE 3 — PPO Fine-Tuning
        # ══════════════════════════════════════════════════════════════
        stage3_label = Text("Stage 3: RL Fine-Tuning (PPO)",
                            font_size=26, color="#3fb950", weight=BOLD, font=_FONT)
        stage3_label.next_to(header, DOWN, buff=0.6)
        self.play(FadeIn(stage3_label, shift=RIGHT * 0.3))

        policy_box = _box("RL Policy\nπ_φ^RL", "#3fb950", width=2.4, height=1.0)
        rm_box2 = _box("Reward Model\nr_θ", "#f0883e", width=2.4, height=1.0)
        sft_box2 = _box("SFT Policy\nπ^SFT", "#d2a8ff", width=2.4, height=1.0)

        policy_box.move_to(ORIGIN + UP * 0.3)
        rm_box2.move_to(ORIGIN + RIGHT * 3.5 + UP * 0.3)
        sft_box2.move_to(ORIGIN + LEFT * 3.5 + UP * 0.3)

        self.play(
            FadeIn(policy_box), FadeIn(rm_box2), FadeIn(sft_box2),
            run_time=0.8
        )

        # Arrows: policy → RM (get reward), SFT → policy (KL penalty)
        a5 = Arrow(policy_box.get_right(), rm_box2.get_left(), buff=0.15,
                    color="#f0883e", stroke_width=2, max_tip_length_to_length_ratio=0.12)
        a6 = Arrow(sft_box2.get_right(), policy_box.get_left(), buff=0.15,
                    color="#d2a8ff", stroke_width=2, max_tip_length_to_length_ratio=0.12)

        lbl5 = _small_label("reward\nsignal", "#f0883e")
        lbl5.next_to(a5, UP, buff=0.15)
        lbl6 = _small_label("KL\npenalty", "#d2a8ff")
        lbl6.next_to(a6, UP, buff=0.15)

        self.play(Create(a5), FadeIn(lbl5), Create(a6), FadeIn(lbl6), run_time=0.8)

        # Show the reward equation
        reward_eq = MathTex(
            r"R(x,y) = r_\theta(x,y) - \beta\,\log\frac{\pi_\phi^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)}",
            font_size=30, color="#3fb950"
        )
        reward_eq.next_to(policy_box, DOWN, buff=0.8)
        self.play(Write(reward_eq), run_time=1.0)
        self.wait(1.5)

        # ── Final summary ──
        stage3_all = VGroup(stage3_label, policy_box, rm_box2, sft_box2, a5, a6, lbl5, lbl6, reward_eq)
        self.play(FadeOut(stage3_all), run_time=0.5)

        # Summary slide
        summary_title = Text("The Complete Pipeline", font_size=30,
                              weight=BOLD, color=WHITE, font=_FONT)
        summary_title.next_to(header, DOWN, buff=0.5)

        steps = VGroup(
            Text("1. SFT on Reddit TL;DR → base policy", font_size=22,
                 color="#58a6ff", font=_FONT),
            Text("2. Train reward model from 65K human comparisons", font_size=22,
                 color="#f0883e", font=_FONT),
            Text("3. PPO with KL-penalized reward → final policy", font_size=22,
                 color="#3fb950", font=_FONT),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        steps.next_to(summary_title, DOWN, buff=0.5)

        result = Text("Result: preferred over human-written summaries",
                       font_size=22, color="#d2a8ff", font=_FONT)
        result.next_to(steps, DOWN, buff=0.5)

        self.play(FadeIn(summary_title), run_time=0.5)
        for step in steps:
            self.play(FadeIn(step, shift=RIGHT * 0.3), run_time=0.5)
        self.play(FadeIn(result, shift=UP * 0.2), run_time=0.6)
        self.wait(2.0)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
