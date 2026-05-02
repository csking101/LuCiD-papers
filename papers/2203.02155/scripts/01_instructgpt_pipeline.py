# 01 — InstructGPT Pipeline
#
# Tool: Manim
# Output: MP4
#
# Animated walkthrough of the 3-step InstructGPT pipeline:
# Step 1: SFT on labeler demonstrations (13K prompts)
# Step 2: Reward Model from K-way rankings (33K prompts)
# Step 3: PPO-ptx with pretraining gradient mix
#
# Run:
#   manim -qm --media_dir ../output/animations 01_instructgpt_pipeline.py InstructGPTPipeline

from manim import *
import atexit
import shutil
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "2203.02155"

_FONT = "Latin Modern Roman"
_SUB = "#c9d1d9"


def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/01_instructgpt_pipeline/720p30/InstructGPTPipeline.mp4"
    dst = _DOCS_DIR / "InstructGPTPipeline.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


def _box(label, color, width=2.6, height=1.0):
    rect = RoundedRectangle(
        corner_radius=0.15, width=width, height=height,
        stroke_color=color, fill_color=color, fill_opacity=0.15,
        stroke_width=2
    )
    txt = Text(label, font_size=20, color=color, font=_FONT)
    txt.move_to(rect.get_center())
    return VGroup(rect, txt)


def _arrow_between(a, b, color=WHITE):
    return Arrow(
        a.get_right(), b.get_left(),
        buff=0.15, color=color, stroke_width=2, max_tip_length_to_length_ratio=0.15
    )


def _label(text, color=_SUB, size=18):
    return Text(text, font_size=size, color=color, font=_FONT)


class InstructGPTPipeline(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("InstructGPT: Training Pipeline", font_size=36,
                      weight=BOLD, color=WHITE, font=_FONT)
        sub = Text("Ouyang et al., 2022 — RLHF at Scale",
                    font_size=22, color=_SUB, font=_FONT)
        header = VGroup(title, sub).arrange(DOWN, buff=0.25).to_edge(UP, buff=0.5)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2), run_time=1.2)
        self.wait(0.6)

        # ══════════════════════════════════════════════════════════════
        # STEP 1 — Supervised Fine-Tuning
        # ══════════════════════════════════════════════════════════════
        s1_label = Text("Step 1: Supervised Fine-Tuning (SFT)",
                        font_size=26, color="#58a6ff", weight=BOLD, font=_FONT)
        s1_label.next_to(header, DOWN, buff=0.6)
        self.play(FadeIn(s1_label, shift=RIGHT * 0.3))

        prompts_box = _box("Labeler Demos\n13K prompts", "#58a6ff", width=2.8, height=1.0)
        gpt3_box = _box("GPT-3\npretrained", "#d2a8ff", width=2.4, height=1.0)
        sft_box = _box("SFT Model", "#3fb950", width=2.4, height=1.0)

        row1 = VGroup(prompts_box, gpt3_box, sft_box).arrange(RIGHT, buff=1.2)
        row1.next_to(s1_label, DOWN, buff=0.5)

        a1 = _arrow_between(prompts_box, gpt3_box, "#58a6ff")
        a2 = _arrow_between(gpt3_box, sft_box, "#d2a8ff")
        lbl1 = _label("supervised\nlearning")
        lbl1.next_to(a2, UP, buff=0.15)

        self.play(FadeIn(prompts_box, shift=UP * 0.3), FadeIn(gpt3_box, shift=UP * 0.3), run_time=0.8)
        self.play(Create(a1), run_time=0.5)
        self.play(Create(a2), FadeIn(lbl1), run_time=0.6)
        self.play(FadeIn(sft_box, shift=UP * 0.3), run_time=0.6)

        detail1 = _label("16 epochs · cosine LR · dropout 0.2", size=16)
        detail1.next_to(row1, DOWN, buff=0.4)
        self.play(FadeIn(detail1), run_time=0.5)
        self.wait(1.0)

        s1_all = VGroup(s1_label, prompts_box, gpt3_box, sft_box, a1, a2, lbl1, detail1)
        self.play(FadeOut(s1_all), run_time=0.5)

        # ══════════════════════════════════════════════════════════════
        # STEP 2 — Reward Model
        # ══════════════════════════════════════════════════════════════
        s2_label = Text("Step 2: Reward Model (RM) Training",
                        font_size=26, color="#f0883e", weight=BOLD, font=_FONT)
        s2_label.next_to(header, DOWN, buff=0.6)
        self.play(FadeIn(s2_label, shift=RIGHT * 0.3))

        outputs_box = _box("K=4..9 Outputs\nper prompt", "#58a6ff", width=2.8, height=1.0)
        labeler_box = _box("Labeler Rankings\n33K prompts", "#f0883e", width=2.8, height=1.0)
        rm_box = _box("6B Reward Model\nr_θ(x, y)", "#f0883e", width=2.8, height=1.0)

        row2 = VGroup(outputs_box, labeler_box, rm_box).arrange(RIGHT, buff=0.9)
        row2.next_to(s2_label, DOWN, buff=0.5)

        a3 = _arrow_between(outputs_box, labeler_box, "#58a6ff")
        a4 = _arrow_between(labeler_box, rm_box, "#f0883e")
        lbl3 = _label("rank K outputs")
        lbl3.next_to(a3, UP, buff=0.15)
        lbl4 = _label("Bradley-Terry\nbatched C(K,2)")
        lbl4.next_to(a4, UP, buff=0.15)

        self.play(FadeIn(outputs_box, shift=UP * 0.3), run_time=0.6)
        self.play(Create(a3), FadeIn(lbl3), FadeIn(labeler_box, shift=UP * 0.3), run_time=0.8)
        self.play(Create(a4), FadeIn(lbl4), run_time=0.6)
        self.play(FadeIn(rm_box, shift=UP * 0.3), run_time=0.6)

        # Show the loss
        loss_eq = MathTex(
            r"\text{loss}(\theta) = -\frac{1}{\binom{K}{2}}"
            r"\mathbb{E}\!\left[\log\sigma\!\left(r_\theta(x,y_w) - r_\theta(x,y_l)\right)\right]",
            font_size=28, color="#f0883e"
        )
        loss_eq.next_to(row2, DOWN, buff=0.5)
        self.play(Write(loss_eq), run_time=1.0)
        self.wait(1.2)

        s2_all = VGroup(s2_label, outputs_box, labeler_box, rm_box, a3, a4, lbl3, lbl4, loss_eq)
        self.play(FadeOut(s2_all), run_time=0.5)

        # ══════════════════════════════════════════════════════════════
        # STEP 3 — PPO-ptx
        # ══════════════════════════════════════════════════════════════
        s3_label = Text("Step 3: RL Fine-Tuning (PPO-ptx)",
                        font_size=26, color="#3fb950", weight=BOLD, font=_FONT)
        s3_label.next_to(header, DOWN, buff=0.6)
        self.play(FadeIn(s3_label, shift=RIGHT * 0.3))

        policy_box = _box("RL Policy\nπ_φ^RL", "#3fb950", width=2.4, height=1.0)
        rm_box2 = _box("Reward Model\nr_θ", "#f0883e", width=2.4, height=1.0)
        sft_box2 = _box("SFT Policy\nπ^SFT", "#d2a8ff", width=2.4, height=1.0)
        pretrain_box = _box("Pretrain Data\nD_pretrain", "#58a6ff", width=2.4, height=1.0)

        # Layout: SFT left, Policy center-top, RM right, Pretrain below
        policy_box.move_to(ORIGIN + UP * 0.5)
        rm_box2.move_to(ORIGIN + RIGHT * 3.5 + UP * 0.5)
        sft_box2.move_to(ORIGIN + LEFT * 3.5 + UP * 0.5)
        pretrain_box.move_to(ORIGIN + DOWN * 1.5)

        self.play(
            FadeIn(policy_box), FadeIn(rm_box2), FadeIn(sft_box2),
            run_time=0.8
        )

        a5 = Arrow(policy_box.get_right(), rm_box2.get_left(), buff=0.15,
                    color="#f0883e", stroke_width=2, max_tip_length_to_length_ratio=0.12)
        a6 = Arrow(sft_box2.get_right(), policy_box.get_left(), buff=0.15,
                    color="#d2a8ff", stroke_width=2, max_tip_length_to_length_ratio=0.12)

        lbl5 = _label("reward\nsignal", "#f0883e")
        lbl5.next_to(a5, UP, buff=0.15)
        lbl6 = _label("KL penalty\nβ = 0.02", "#d2a8ff")
        lbl6.next_to(a6, UP, buff=0.15)

        self.play(Create(a5), FadeIn(lbl5), Create(a6), FadeIn(lbl6), run_time=0.8)

        # Show pretrain box and arrow
        self.play(FadeIn(pretrain_box, shift=UP * 0.3), run_time=0.5)
        a7 = Arrow(pretrain_box.get_top(), policy_box.get_bottom(), buff=0.15,
                    color="#58a6ff", stroke_width=2, max_tip_length_to_length_ratio=0.12)
        lbl7 = _label("pretraining\ngradients (γ=27.8)", "#58a6ff")
        lbl7.next_to(a7, RIGHT, buff=0.15)
        self.play(Create(a7), FadeIn(lbl7), run_time=0.7)

        # highlight the new contribution
        new_badge = Text("NEW in InstructGPT", font_size=16, color="#3fb950",
                         weight=BOLD, font=_FONT)
        new_rect = SurroundingRectangle(VGroup(pretrain_box, lbl7),
                                         color="#3fb950", buff=0.15, stroke_width=1.5)
        new_badge.next_to(new_rect, DOWN, buff=0.15)
        self.play(Create(new_rect), FadeIn(new_badge), run_time=0.6)
        self.wait(1.5)

        # ── Final summary ──
        s3_all = VGroup(s3_label, policy_box, rm_box2, sft_box2, pretrain_box,
                        a5, a6, a7, lbl5, lbl6, lbl7, new_rect, new_badge)
        self.play(FadeOut(s3_all), run_time=0.5)

        # Summary slide
        sum_title = Text("The InstructGPT Recipe", font_size=30,
                         weight=BOLD, color=WHITE, font=_FONT)
        sum_title.next_to(header, DOWN, buff=0.5)

        steps = VGroup(
            Text("1. SFT on 13K labeler demonstrations", font_size=22,
                 color="#58a6ff", font=_FONT),
            Text("2. Train 6B reward model from K-way rankings (33K)", font_size=22,
                 color="#f0883e", font=_FONT),
            Text("3. PPO-ptx: RL + pretraining gradient mix", font_size=22,
                 color="#3fb950", font=_FONT),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        steps.next_to(sum_title, DOWN, buff=0.5)

        result = Text("1.3B InstructGPT preferred over 175B GPT-3",
                       font_size=22, color="#d2a8ff", font=_FONT)
        result.next_to(steps, DOWN, buff=0.5)

        self.play(FadeIn(sum_title), run_time=0.5)
        for step in steps:
            self.play(FadeIn(step, shift=RIGHT * 0.3), run_time=0.5)
        self.play(FadeIn(result, shift=UP * 0.2), run_time=0.6)
        self.wait(2.0)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
