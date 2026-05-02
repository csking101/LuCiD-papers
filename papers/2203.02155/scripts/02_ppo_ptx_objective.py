# 02 — PPO-ptx Objective Derivation
#
# Tool: Manim
# Output: MP4
#
# Animated step-by-step build-up of the PPO-ptx objective:
# 1. Start from standard RL reward: E[r_θ(x,y)]
# 2. Add KL penalty: - β·log(π_RL/π_SFT)
# 3. Add pretraining gradient: + γ·E[log(π_RL(x))]
# Show why γ=0 causes alignment tax (NLP benchmarks regress).
#
# Run:
#   manim -qm --media_dir ../output/animations 02_ppo_ptx_objective.py PPOptxObjective

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
    src = _PAPER_DIR / "output/animations/videos/02_ppo_ptx_objective/720p30/PPOptxObjective.mp4"
    dst = _DOCS_DIR / "PPOptxObjective.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


class PPOptxObjective(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("The PPO-ptx Objective", font_size=36,
                      weight=BOLD, color=WHITE, font=_FONT)
        sub = Text("Mixing RL reward with pretraining to prevent alignment tax",
                    font_size=22, color=_SUB, font=_FONT)
        header = VGroup(title, sub).arrange(DOWN, buff=0.25).to_edge(UP, buff=0.5)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2), run_time=1.2)
        self.wait(0.5)

        # ══════════════════════════════════════════════════════════════
        # PHASE 1: Standard RL objective
        # ══════════════════════════════════════════════════════════════
        phase1_label = Text("Phase 1: Standard RL Objective",
                            font_size=26, color="#58a6ff", weight=BOLD, font=_FONT)
        phase1_label.next_to(header, DOWN, buff=0.6)
        self.play(FadeIn(phase1_label, shift=RIGHT * 0.3))

        eq1 = MathTex(
            r"\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi^{RL}}}"
            r"\left[ r_\theta(x, y) \right]",
            font_size=34, color=WHITE
        )
        eq1.next_to(phase1_label, DOWN, buff=0.5)

        note1 = Text("Maximize reward from the learned reward model",
                      font_size=20, color=_SUB, font=_FONT)
        note1.next_to(eq1, DOWN, buff=0.4)

        self.play(Write(eq1), run_time=1.2)
        self.play(FadeIn(note1), run_time=0.5)

        problem1 = Text("Problem: Policy can exploit reward model (overoptimization)",
                         font_size=20, color="#f85149", font=_FONT)
        problem1.next_to(note1, DOWN, buff=0.4)
        self.play(FadeIn(problem1, shift=UP * 0.2), run_time=0.6)
        self.wait(1.0)

        p1_group = VGroup(phase1_label, eq1, note1, problem1)
        self.play(FadeOut(p1_group), run_time=0.4)

        # ══════════════════════════════════════════════════════════════
        # PHASE 2: Add KL penalty (from summarization paper)
        # ══════════════════════════════════════════════════════════════
        phase2_label = Text("Phase 2: Add KL Penalty (from Stiennon et al.)",
                            font_size=26, color="#d2a8ff", weight=BOLD, font=_FONT)
        phase2_label.next_to(header, DOWN, buff=0.6)
        self.play(FadeIn(phase2_label, shift=RIGHT * 0.3))

        eq2 = MathTex(
            r"\text{objective}(\phi) = \mathbb{E}\!\left["
            r"r_\theta(x, y)"
            r"- \beta \log \frac{\pi_\phi^{RL}(y|x)}{\pi^{SFT}(y|x)}"
            r"\right]",
            font_size=30, color=WHITE
        )
        eq2.next_to(phase2_label, DOWN, buff=0.5)

        # Highlight the KL term
        kl_brace = Brace(eq2[0][17:38], DOWN, color="#d2a8ff")
        kl_label = Text("KL penalty: stay close to SFT", font_size=18,
                         color="#d2a8ff", font=_FONT)
        kl_label.next_to(kl_brace, DOWN, buff=0.2)

        beta_note = Text("β = 0.02 (per token)", font_size=20, color=_SUB, font=_FONT)
        beta_note.next_to(kl_label, DOWN, buff=0.4)

        self.play(Write(eq2), run_time=1.2)
        self.play(Create(kl_brace), FadeIn(kl_label), run_time=0.7)
        self.play(FadeIn(beta_note), run_time=0.4)

        problem2 = Text("Problem: Policy forgets pretrained capabilities (alignment tax)",
                         font_size=20, color="#f85149", font=_FONT)
        problem2.next_to(beta_note, DOWN, buff=0.4)
        self.play(FadeIn(problem2, shift=UP * 0.2), run_time=0.6)
        self.wait(1.2)

        p2_group = VGroup(phase2_label, eq2, kl_brace, kl_label, beta_note, problem2)
        self.play(FadeOut(p2_group), run_time=0.4)

        # ══════════════════════════════════════════════════════════════
        # PHASE 3: PPO-ptx — add pretraining gradient
        # ══════════════════════════════════════════════════════════════
        phase3_label = Text("Phase 3: PPO-ptx — Mix in Pretraining",
                            font_size=26, color="#3fb950", weight=BOLD, font=_FONT)
        phase3_label.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(phase3_label, shift=RIGHT * 0.3))

        # Full equation on two lines for readability
        eq3_top = MathTex(
            r"\text{objective}(\phi) = \mathbb{E}\!\left["
            r"r_\theta(x, y) - \beta \log \frac{\pi_\phi^{RL}(y|x)}{\pi^{SFT}(y|x)}"
            r"\right]",
            font_size=28, color=WHITE
        )

        eq3_bot = MathTex(
            r"+ \; \gamma \; \mathbb{E}_{x \sim D_{\text{pretrain}}}"
            r"\!\left[ \log \pi_\phi^{RL}(x) \right]",
            font_size=28, color="#3fb950"
        )

        eq3 = VGroup(eq3_top, eq3_bot).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        eq3.next_to(phase3_label, DOWN, buff=0.4)

        self.play(Write(eq3_top), run_time=1.0)
        self.wait(0.3)
        self.play(Write(eq3_bot), run_time=1.0)

        # Brace on the new term
        ptx_brace = Brace(eq3_bot, DOWN, color="#3fb950")
        ptx_label = Text("Pretraining gradient: maintain general capabilities",
                          font_size=18, color="#3fb950", font=_FONT)
        ptx_label.next_to(ptx_brace, DOWN, buff=0.2)

        gamma_note = Text("γ = 27.8 (controls pretraining mix strength)",
                           font_size=20, color=_SUB, font=_FONT)
        gamma_note.next_to(ptx_label, DOWN, buff=0.3)

        self.play(Create(ptx_brace), FadeIn(ptx_label), run_time=0.7)
        self.play(FadeIn(gamma_note), run_time=0.4)

        # Highlight box around the new term
        new_rect = SurroundingRectangle(eq3_bot, color="#3fb950", buff=0.15, stroke_width=2)
        new_badge = Text("NEW: prevents alignment tax", font_size=16,
                         color="#3fb950", weight=BOLD, font=_FONT)
        new_badge.next_to(new_rect, RIGHT, buff=0.2)
        self.play(Create(new_rect), FadeIn(new_badge), run_time=0.6)
        self.wait(1.5)

        p3_group = VGroup(phase3_label, eq3, ptx_brace, ptx_label,
                          gamma_note, new_rect, new_badge)
        self.play(FadeOut(p3_group), run_time=0.4)

        # ══════════════════════════════════════════════════════════════
        # SUMMARY: side-by-side γ=0 vs γ=27.8
        # ══════════════════════════════════════════════════════════════
        sum_title = Text("The Alignment Tax Fix", font_size=30,
                         weight=BOLD, color=WHITE, font=_FONT)
        sum_title.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(sum_title))

        # Two columns
        col_left_title = Text("PPO (γ = 0)", font_size=24, color="#f85149",
                               weight=BOLD, font=_FONT)
        col_right_title = Text("PPO-ptx (γ = 27.8)", font_size=24, color="#3fb950",
                                weight=BOLD, font=_FONT)

        col_left_items = VGroup(
            Text("HellaSwag: 78.6 → 71.4  ↓", font_size=20, color="#f85149", font=_FONT),
            Text("SQuAD F1: 69.0 → 49.2  ↓", font_size=20, color="#f85149", font=_FONT),
            Text("DROP F1: 36.7 → 24.0  ↓", font_size=20, color="#f85149", font=_FONT),
            Text("FR→EN: 32.6 → 18.4  ↓", font_size=20, color="#f85149", font=_FONT),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        col_right_items = VGroup(
            Text("HellaSwag: 78.6 → 78.8  ✓", font_size=20, color="#3fb950", font=_FONT),
            Text("SQuAD F1: 69.0 → 65.8  ~", font_size=20, color="#3fb950", font=_FONT),
            Text("DROP F1: 36.7 → 36.5  ✓", font_size=20, color="#3fb950", font=_FONT),
            Text("FR→EN: 32.6 → 33.8  ✓", font_size=20, color="#3fb950", font=_FONT),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        col_left = VGroup(col_left_title, col_left_items).arrange(DOWN, buff=0.3)
        col_right = VGroup(col_right_title, col_right_items).arrange(DOWN, buff=0.3)

        cols = VGroup(col_left, col_right).arrange(RIGHT, buff=1.0)
        cols.next_to(sum_title, DOWN, buff=0.5)

        divider = Line(cols.get_top() + UP * 0.1, cols.get_bottom() + DOWN * 0.1,
                        color=_SUB, stroke_width=1)

        self.play(FadeIn(col_left_title), FadeIn(col_right_title), Create(divider), run_time=0.6)
        for l_item, r_item in zip(col_left_items, col_right_items):
            self.play(FadeIn(l_item, shift=RIGHT * 0.2), FadeIn(r_item, shift=LEFT * 0.2),
                      run_time=0.5)
        self.wait(2.0)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
