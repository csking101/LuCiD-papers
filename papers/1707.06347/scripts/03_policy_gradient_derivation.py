# 03 — Policy Gradient Derivation
#
# Tool: Manim
# Output: MP4
#
# Step-by-step animation of the policy gradient derivation:
# grad(E[R]) → log-derivative trick → trajectory simplification
# → REINFORCE estimator → advantage function
#
# Run:
#   manim -qm --media_dir ../output/animations 03_policy_gradient_derivation.py PolicyGradientDerivation

from manim import *
import atexit
import shutil
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "1707.06347"

# Font for descriptive text — matches LaTeX aesthetic
_FONT = "Latin Modern Roman"
_SUB = "#c9d1d9"  # high-contrast secondary text


def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/03_policy_gradient_derivation/720p30/PolicyGradientDerivation.mp4"
    dst = _DOCS_DIR / "PolicyGradientDerivation.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


class PolicyGradientDerivation(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("Deriving the Policy Gradient", font_size=36, weight=BOLD,
                      color=WHITE, font=_FONT)
        subtitle = Text("From objective to estimator, step by step", font_size=22,
                         color=_SUB, font=_FONT)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ════════════════════════════════════════
        # STEP 1: The Objective
        # ════════════════════════════════════════
        s1_label = Text("Step 1: The Objective", font_size=28, weight=BOLD,
                         color="#58a6ff", font=_FONT)
        s1_label.to_edge(UP, buff=0.4)
        self.play(Write(s1_label), run_time=0.5)

        obj = MathTex(
            r"J(\theta) = \mathbb{E}\!\left[\sum_t r_t\right]",
            font_size=38, color=WHITE,
        )
        obj.next_to(s1_label, DOWN, buff=0.6)
        self.play(Write(obj), run_time=0.8)

        obj_expand = MathTex(
            r"= \int P(\tau \mid \theta) \, R(\tau) \, d\tau",
            font_size=34, color=_SUB,
        )
        obj_expand.next_to(obj, DOWN, buff=0.3)
        self.play(Write(obj_expand), run_time=0.8)

        note1 = MathTex(
            r"\text{We want: } \nabla_\theta J(\theta)",
            font_size=34, color="#f0883e",
        )
        note1.next_to(obj_expand, DOWN, buff=0.5)
        self.play(FadeIn(note1))

        problem = Text("Problem: gradient of an integral over trajectories",
                        font_size=20, color="#E74C3C", font=_FONT)
        problem.next_to(note1, DOWN, buff=0.3)
        self.play(FadeIn(problem))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # STEP 2: The Log-Derivative Trick
        # ════════════════════════════════════════
        s2_label = Text("Step 2: The Log-Derivative Trick", font_size=28, weight=BOLD,
                         color="#58a6ff", font=_FONT)
        s2_label.to_edge(UP, buff=0.4)
        self.play(Write(s2_label), run_time=0.5)

        identity = MathTex(
            r"\nabla f = f \cdot \nabla(\log f)",
            font_size=38, color="#3fb950",
        )
        identity.next_to(s2_label, DOWN, buff=0.5)
        self.play(Write(identity), run_time=0.8)

        because = MathTex(
            r"\text{Because } \nabla(\log f) = \frac{\nabla f}{f}",
            font_size=30, color=_SUB,
        )
        because.next_to(identity, DOWN, buff=0.2)
        self.play(FadeIn(because))
        self.wait(0.5)

        apply_text = MathTex(
            r"\text{Apply to } P(\tau \mid \theta)\text{:}",
            font_size=30, color=WHITE,
        )
        apply_text.next_to(because, DOWN, buff=0.5)
        self.play(FadeIn(apply_text))

        result = MathTex(
            r"\nabla P(\tau|\theta) = P(\tau|\theta) \,\nabla \log P(\tau|\theta)",
            font_size=32, color=WHITE,
        )
        result.next_to(apply_text, DOWN, buff=0.3)
        self.play(Write(result), run_time=0.8)
        self.wait(0.5)

        magic = Text("Now the gradient moves INSIDE the expectation!",
                      font_size=22, weight=BOLD, color="#f0883e", font=_FONT)
        magic.next_to(result, DOWN, buff=0.5)
        self.play(FadeIn(magic))

        new_form = MathTex(
            r"\nabla J = \mathbb{E}\!\left[\nabla \log P(\tau|\theta) \cdot R(\tau)\right]",
            font_size=36, color="#3fb950",
        )
        new_form.next_to(magic, DOWN, buff=0.3)
        self.play(Write(new_form), run_time=0.8)
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # STEP 3: Simplify log P(tau)
        # ════════════════════════════════════════
        s3_label = MathTex(
            r"\text{Step 3: Simplify } \log P(\tau|\theta)",
            font_size=34, color="#58a6ff",
        )
        s3_label.to_edge(UP, buff=0.4)
        self.play(Write(s3_label), run_time=0.5)

        traj_prob = MathTex(
            r"P(\tau|\theta) = p(s_0) \prod_t \pi_\theta(a_t|s_t) \, P(s_{t+1}|s_t,a_t)",
            font_size=28, color=WHITE,
        )
        traj_prob.next_to(s3_label, DOWN, buff=0.5)
        self.play(Write(traj_prob), run_time=0.8)

        log_form = MathTex(
            r"\log P(\tau|\theta) = \log p(s_0) + \sum_t \log \pi_\theta(a_t|s_t)"
            r" + \sum_t \log P(s_{t+1}|\cdots)",
            font_size=24, color=WHITE,
        )
        log_form.next_to(traj_prob, DOWN, buff=0.3)
        self.play(Write(log_form), run_time=0.8)
        self.wait(0.5)

        # Highlight what survives
        grad_text = MathTex(
            r"\text{Take } \nabla_\theta \text{:}",
            font_size=30, color="#f0883e",
        )
        grad_text.next_to(log_form, DOWN, buff=0.5)
        self.play(FadeIn(grad_text))

        vanish1 = MathTex(
            r"\log p(s_0) \;\longrightarrow\; \text{vanishes (no } \theta \text{)}",
            font_size=26, color="#E74C3C",
        )
        vanish2 = MathTex(
            r"\log P(s_{t+1}|\cdots) \;\longrightarrow\; \text{vanishes (env dynamics)}",
            font_size=26, color="#E74C3C",
        )
        survives = MathTex(
            r"\sum_t \log \pi_\theta(a_t|s_t) \;\longrightarrow\; \textbf{SURVIVES}",
            font_size=28, color="#3fb950",
        )
        VGroup(vanish1, vanish2, survives).arrange(DOWN, buff=0.2).next_to(grad_text, DOWN, buff=0.3)

        self.play(FadeIn(vanish1), run_time=0.4)
        self.play(FadeIn(vanish2), run_time=0.4)
        self.play(FadeIn(survives), run_time=0.4)
        self.wait(0.5)

        model_free = Text("This is why policy gradients are MODEL-FREE",
                           font_size=20, color="#d2a8ff", font=_FONT)
        model_free.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(model_free))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # STEP 4: The REINFORCE Estimator
        # ════════════════════════════════════════
        s4_label = Text("Step 4: The REINFORCE Estimator", font_size=28, weight=BOLD,
                         color="#58a6ff", font=_FONT)
        s4_label.to_edge(UP, buff=0.4)
        self.play(Write(s4_label), run_time=0.5)

        reinforce = MathTex(
            r"\nabla J = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]",
            font_size=34, color=WHITE,
        )
        reinforce.next_to(s4_label, DOWN, buff=0.6)
        self.play(Write(reinforce), run_time=0.8)

        plain = Text("In plain English:", font_size=20, color=_SUB, font=_FONT)
        plain.next_to(reinforce, DOWN, buff=0.4)
        self.play(FadeIn(plain))

        bullets = VGroup(
            VGroup(
                MathTex(r"\nabla \log \pi", font_size=28, color="#58a6ff"),
                Text("  =  direction to make action more likely", font_size=18,
                     color="#58a6ff", font=_FONT),
            ).arrange(RIGHT, buff=0.1),
            VGroup(
                MathTex(r"R(\tau)", font_size=28, color="#3fb950"),
                Text("  =  how good was this trajectory", font_size=18,
                     color="#3fb950", font=_FONT),
            ).arrange(RIGHT, buff=0.1),
            Text("Product  =  push good actions up, bad actions down",
                 font_size=18, color="#f0883e", font=_FONT),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        bullets.next_to(plain, DOWN, buff=0.3)

        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.4)
        self.wait(1)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # STEP 5: Replace R(tau) with Advantage
        # ════════════════════════════════════════
        s5_label = Text("Step 5: Use Advantage for Lower Variance", font_size=28,
                         weight=BOLD, color="#58a6ff", font=_FONT)
        s5_label.to_edge(UP, buff=0.4)
        self.play(Write(s5_label), run_time=0.5)

        before_eq = MathTex(
            r"\text{Before: } \nabla J = \mathbb{E}\!\left[\nabla \log \pi \cdot R(\tau)\right]",
            font_size=30, color=_SUB,
        )
        before_eq.next_to(s5_label, DOWN, buff=0.5)
        self.play(Write(before_eq), run_time=0.6)

        problem_text = MathTex(
            r"\text{Problem: } R(\tau) \text{ is always positive} \;\Rightarrow\; \text{high variance}",
            font_size=28, color="#E74C3C",
        )
        problem_text.next_to(before_eq, DOWN, buff=0.3)
        self.play(FadeIn(problem_text))
        self.wait(0.5)

        solution = MathTex(
            r"\text{Solution: subtract baseline } V(s)",
            font_size=30, color="#3fb950",
        )
        solution.next_to(problem_text, DOWN, buff=0.4)
        self.play(FadeIn(solution))

        adv_def = MathTex(
            r"\hat{A}(s,a) = Q(s,a) - V(s)",
            font_size=36, color=WHITE,
        )
        adv_label = Text("(advantage)", font_size=20, color=_SUB, font=_FONT)
        adv_label.next_to(adv_def, RIGHT, buff=0.2)
        adv_group = VGroup(adv_def, adv_label)
        adv_group.next_to(solution, DOWN, buff=0.3)
        self.play(Write(adv_def), FadeIn(adv_label), run_time=0.7)
        self.wait(0.5)

        after_eq = MathTex(
            r"\text{After: } \nabla J = \mathbb{E}\!\left[\nabla \log \pi \cdot \hat{A}(s,a)\right]",
            font_size=32, color="#3fb950",
        )
        after_eq.next_to(adv_group, DOWN, buff=0.4)
        self.play(Write(after_eq), run_time=0.8)

        same_expect = Text("Same expected gradient, much lower variance!",
                            font_size=20, color="#f0883e", font=_FONT)
        same_expect.next_to(after_eq, DOWN, buff=0.4)
        self.play(FadeIn(same_expect))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # FINAL: The Policy Gradient Estimator
        # ════════════════════════════════════════
        final_label = Text("The Policy Gradient Estimator", font_size=30, weight=BOLD,
                            color="#58a6ff", font=_FONT)
        final_label.to_edge(UP, buff=0.5)
        self.play(Write(final_label), run_time=0.6)

        final_eq = MathTex(
            r"\hat{g} = \hat{\mathbb{E}}_t\!\left["
            r"\nabla_\theta \log \pi_\theta(a_t \mid s_t) \,\hat{A}_t\right]",
            font_size=40, color=WHITE,
        )
        final_eq.move_to(ORIGIN)
        box = SurroundingRectangle(final_eq, color="#f0883e", buff=0.25, corner_radius=0.1)

        self.play(Write(final_eq), run_time=0.8)
        self.play(Create(box), run_time=0.5)

        footnote = Text("This is Equation 1 in the PPO paper", font_size=18,
                         color=_SUB, font=_FONT)
        footnote.next_to(box, DOWN, buff=0.4)
        self.play(FadeIn(footnote))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])
