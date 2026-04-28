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
#   manim -ql --media_dir ../output/animations 03_policy_gradient_derivation.py PolicyGradientDerivation

from manim import *
import atexit
import shutil
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "1707.06347"


def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/03_policy_gradient_derivation/480p15/PolicyGradientDerivation.mp4"
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
        title = Text("Deriving the Policy Gradient", font_size=36, weight=BOLD, color=WHITE)
        subtitle = Text("From objective to estimator, step by step", font_size=20, color=GREY)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ════════════════════════════════════════
        # STEP 1: The Objective
        # ════════════════════════════════════════
        s1_label = Text("Step 1: The Objective", font_size=26, weight=BOLD, color="#58a6ff")
        s1_label.to_edge(UP, buff=0.4)
        self.play(Write(s1_label), run_time=0.5)

        obj = Text("J(theta) = E [ sum of rewards ]", font_size=22, color=WHITE)
        obj.next_to(s1_label, DOWN, buff=0.6)
        self.play(Write(obj), run_time=0.8)

        obj_expand = Text("= integral of  P(tau|theta) * R(tau)  d_tau", font_size=20, color=GREY)
        obj_expand.next_to(obj, DOWN, buff=0.3)
        self.play(Write(obj_expand), run_time=0.8)

        note1 = Text("We want: grad_theta J(theta)", font_size=20, color="#f0883e")
        note1.next_to(obj_expand, DOWN, buff=0.5)
        self.play(FadeIn(note1))

        problem = Text("Problem: gradient of an integral over trajectories", font_size=18, color="#E74C3C")
        problem.next_to(note1, DOWN, buff=0.3)
        self.play(FadeIn(problem))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # STEP 2: The Log-Derivative Trick
        # ════════════════════════════════════════
        s2_label = Text("Step 2: The Log-Derivative Trick", font_size=26, weight=BOLD, color="#58a6ff")
        s2_label.to_edge(UP, buff=0.4)
        self.play(Write(s2_label), run_time=0.5)

        identity = Text("Identity:  grad f = f * grad(log f)", font_size=22, color="#3fb950")
        identity.next_to(s2_label, DOWN, buff=0.5)
        self.play(Write(identity), run_time=0.8)

        because = Text("Because:  grad(log f) = grad(f) / f", font_size=18, color=GREY)
        because.next_to(identity, DOWN, buff=0.2)
        self.play(FadeIn(because))
        self.wait(0.5)

        apply_text = Text("Apply to P(tau|theta):", font_size=20, color=WHITE)
        apply_text.next_to(because, DOWN, buff=0.5)
        self.play(FadeIn(apply_text))

        result = Text("grad P(tau|theta) = P(tau|theta) * grad log P(tau|theta)", font_size=18, color=WHITE)
        result.next_to(apply_text, DOWN, buff=0.3)
        self.play(Write(result), run_time=0.8)
        self.wait(0.5)

        magic = Text("Now the gradient moves INSIDE the expectation!", font_size=20, weight=BOLD, color="#f0883e")
        magic.next_to(result, DOWN, buff=0.5)
        self.play(FadeIn(magic))

        new_form = Text("grad J = E [ grad(log P(tau|theta)) * R(tau) ]", font_size=22, color="#3fb950")
        new_form.next_to(magic, DOWN, buff=0.3)
        self.play(Write(new_form), run_time=0.8)
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # STEP 3: Simplify log P(tau)
        # ════════════════════════════════════════
        s3_label = Text("Step 3: Simplify log P(tau|theta)", font_size=26, weight=BOLD, color="#58a6ff")
        s3_label.to_edge(UP, buff=0.4)
        self.play(Write(s3_label), run_time=0.5)

        traj_prob = Text("P(tau|theta) = p(s0) * prod[ pi(a_t|s_t) * P(s_t+1|s_t,a_t) ]",
                          font_size=18, color=WHITE)
        traj_prob.next_to(s3_label, DOWN, buff=0.5)
        self.play(Write(traj_prob), run_time=0.8)

        log_form = Text("log P(tau|theta) = log p(s0) + sum log pi(a_t|s_t) + sum log P(s_t+1|...)",
                          font_size=16, color=WHITE)
        log_form.next_to(traj_prob, DOWN, buff=0.3)
        self.play(Write(log_form), run_time=0.8)
        self.wait(0.5)

        # Highlight what survives
        grad_text = Text("Take gradient w.r.t. theta:", font_size=20, color="#f0883e")
        grad_text.next_to(log_form, DOWN, buff=0.5)
        self.play(FadeIn(grad_text))

        vanish1 = Text("log p(s0)  -->  vanishes (no theta)", font_size=16, color="#E74C3C")
        vanish2 = Text("log P(s_t+1|...)  -->  vanishes (environment dynamics)", font_size=16, color="#E74C3C")
        survives = Text("sum log pi(a_t|s_t)  -->  SURVIVES", font_size=18, weight=BOLD, color="#3fb950")
        VGroup(vanish1, vanish2, survives).arrange(DOWN, buff=0.2).next_to(grad_text, DOWN, buff=0.3)

        self.play(FadeIn(vanish1), run_time=0.4)
        self.play(FadeIn(vanish2), run_time=0.4)
        self.play(FadeIn(survives), run_time=0.4)
        self.wait(0.5)

        model_free = Text("This is why policy gradients are MODEL-FREE", font_size=18, color="#d2a8ff")
        model_free.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(model_free))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # STEP 4: The REINFORCE Estimator
        # ════════════════════════════════════════
        s4_label = Text("Step 4: The REINFORCE Estimator", font_size=26, weight=BOLD, color="#58a6ff")
        s4_label.to_edge(UP, buff=0.4)
        self.play(Write(s4_label), run_time=0.5)

        reinforce = Text("grad J = E [ sum grad(log pi(a_t|s_t)) * R(tau) ]",
                           font_size=22, color=WHITE)
        reinforce.next_to(s4_label, DOWN, buff=0.6)
        self.play(Write(reinforce), run_time=0.8)

        plain = Text("In plain English:", font_size=18, color=GREY)
        plain.next_to(reinforce, DOWN, buff=0.4)
        self.play(FadeIn(plain))

        bullets = VGroup(
            Text("grad(log pi)  =  direction to make action more likely", font_size=16, color="#58a6ff"),
            Text("R(tau)  =  how good was this trajectory", font_size=16, color="#3fb950"),
            Text("Product  =  push good actions up, bad actions down", font_size=16, color="#f0883e"),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bullets.next_to(plain, DOWN, buff=0.3)

        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.4)
        self.wait(1)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # STEP 5: Replace R(tau) with Advantage
        # ════════════════════════════════════════
        s5_label = Text("Step 5: Use Advantage for Lower Variance", font_size=26, weight=BOLD, color="#58a6ff")
        s5_label.to_edge(UP, buff=0.4)
        self.play(Write(s5_label), run_time=0.5)

        before_eq = Text("Before:  grad J = E [ grad(log pi) * R(tau) ]", font_size=20, color=GREY)
        before_eq.next_to(s5_label, DOWN, buff=0.5)
        self.play(Write(before_eq), run_time=0.6)

        problem_text = Text("Problem: R(tau) is always positive -> high variance",
                             font_size=18, color="#E74C3C")
        problem_text.next_to(before_eq, DOWN, buff=0.3)
        self.play(FadeIn(problem_text))
        self.wait(0.5)

        solution = Text("Solution: subtract baseline V(s)", font_size=20, color="#3fb950")
        solution.next_to(problem_text, DOWN, buff=0.4)
        self.play(FadeIn(solution))

        adv_def = Text("A(s,a) = Q(s,a) - V(s)  (advantage)", font_size=22, color=WHITE)
        adv_def.next_to(solution, DOWN, buff=0.3)
        self.play(Write(adv_def), run_time=0.7)
        self.wait(0.5)

        after_eq = Text("After:  grad J = E [ grad(log pi) * A(s,a) ]", font_size=22, color="#3fb950")
        after_eq.next_to(adv_def, DOWN, buff=0.4)
        self.play(Write(after_eq), run_time=0.8)

        same_expect = Text("Same expected gradient, much lower variance!", font_size=18, color="#f0883e")
        same_expect.next_to(after_eq, DOWN, buff=0.4)
        self.play(FadeIn(same_expect))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # FINAL: The PPO Estimator
        # ════════════════════════════════════════
        final_label = Text("The Policy Gradient Estimator", font_size=28, weight=BOLD, color="#58a6ff")
        final_label.to_edge(UP, buff=0.5)
        self.play(Write(final_label), run_time=0.6)

        final_eq = Text("g = E_t [ grad(log pi(a_t|s_t)) * A_t ]", font_size=26, color=WHITE)
        final_eq.move_to(ORIGIN)
        box = SurroundingRectangle(final_eq, color="#f0883e", buff=0.25, corner_radius=0.1)

        self.play(Write(final_eq), run_time=0.8)
        self.play(Create(box), run_time=0.5)

        footnote = Text("This is Equation 1 in the PPO paper", font_size=16, color=GREY)
        footnote.next_to(box, DOWN, buff=0.4)
        self.play(FadeIn(footnote))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])
