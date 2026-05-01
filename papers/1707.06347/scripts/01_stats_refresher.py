# 01 — Statistics Refresher for PPO
#
# Tool: Manim
# Output: MP4
#
# Animated primer on key math concepts needed to understand PPO:
# 1. Expectation (integral definition → sampling)
# 2. Variance (why it matters for gradient estimates)
# 3. Importance Sampling (evaluating one distribution using samples from another)
# 4. KL Divergence (measuring distance between distributions)
#
# Run:
#   manim -qm --media_dir ../output/animations 01_stats_refresher.py StatsRefresher

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
    src = _PAPER_DIR / "output/animations/videos/01_stats_refresher/720p30/StatsRefresher.mp4"
    dst = _DOCS_DIR / "StatsRefresher.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


class StatsRefresher(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("Statistics Refresher for PPO", font_size=36, weight=BOLD,
                      color=WHITE, font=_FONT)
        subtitle = Text("Four key concepts you need", font_size=22, color=_SUB, font=_FONT)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.6)
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ════════════════════════════════════════
        # 1. EXPECTATION
        # ════════════════════════════════════════
        sec1 = Text("1. Expectation", font_size=32, weight=BOLD, color="#58a6ff", font=_FONT)
        sec1.to_edge(UP, buff=0.5)
        self.play(Write(sec1), run_time=0.7)

        eq_title = Text("The weighted average of outcomes", font_size=22, color=_SUB, font=_FONT)
        eq_title.next_to(sec1, DOWN, buff=0.4)
        self.play(FadeIn(eq_title))

        # Discrete case — proper LaTeX
        disc = MathTex(
            r"\mathbb{E}[f(x)] = \sum_x p(x) \, f(x)",
            font_size=36, color=WHITE,
        )
        disc.next_to(eq_title, DOWN, buff=0.5)
        self.play(Write(disc), run_time=0.8)
        self.wait(0.5)

        # Continuous case — proper LaTeX
        cont = MathTex(
            r"\mathbb{E}[f(x)] = \int p(x) \, f(x) \, dx",
            font_size=36, color=WHITE,
        )
        cont.next_to(disc, DOWN, buff=0.3)
        self.play(Write(cont), run_time=0.8)
        self.wait(0.5)

        # Key insight
        key1 = Text("In RL: we can't compute the integral,", font_size=20, color="#f0883e", font=_FONT)
        key1b = Text("so we estimate by sampling trajectories", font_size=20, color="#f0883e", font=_FONT)
        key1.next_to(cont, DOWN, buff=0.5)
        key1b.next_to(key1, DOWN, buff=0.2)
        self.play(FadeIn(key1), FadeIn(key1b))

        # Sample average
        sample_label = Text("Sample average approximates expectation:",
                            font_size=20, color=_SUB, font=_FONT)
        sample_label.next_to(key1b, DOWN, buff=0.4)
        self.play(FadeIn(sample_label))

        approx = MathTex(
            r"\mathbb{E}[f(x)] \;\approx\; \frac{1}{N} \sum_{i=1}^{N} f(x_i)",
            font_size=36, color="#3fb950",
        )
        approx.next_to(sample_label, DOWN, buff=0.3)
        self.play(Write(approx), run_time=0.8)
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects if m != sec1])
        self.play(FadeOut(sec1))

        # ════════════════════════════════════════
        # 2. VARIANCE
        # ════════════════════════════════════════
        sec2 = Text("2. Variance", font_size=32, weight=BOLD, color="#58a6ff", font=_FONT)
        sec2.to_edge(UP, buff=0.5)
        self.play(Write(sec2), run_time=0.7)

        var_def = MathTex(
            r"\text{Var}[X] = \mathbb{E}\!\left[(X - \mathbb{E}[X])^2\right]",
            font_size=36, color=WHITE,
        )
        var_def.next_to(sec2, DOWN, buff=0.5)
        self.play(Write(var_def), run_time=0.8)

        meaning = Text("How spread out are our estimates?", font_size=22, color=_SUB, font=_FONT)
        meaning.next_to(var_def, DOWN, buff=0.4)
        self.play(FadeIn(meaning))
        self.wait(0.5)

        # Low variance visual
        low_var_label = Text("Low Variance: estimates cluster tightly",
                             font_size=20, color="#3fb950", font=_FONT)
        low_var_label.move_to(DOWN * 0.5 + LEFT * 3)

        low_dots = VGroup()
        center_low = DOWN * 1.5 + LEFT * 3
        for i in range(8):
            d = Dot(center_low + RIGHT * (i - 3.5) * 0.12 + UP * ((i % 3) - 1) * 0.08,
                     radius=0.06, color="#3fb950")
            low_dots.add(d)

        # High variance visual
        high_var_label = Text("High Variance: estimates all over",
                              font_size=20, color="#E74C3C", font=_FONT)
        high_var_label.move_to(DOWN * 0.5 + RIGHT * 3)

        high_dots = VGroup()
        center_high = DOWN * 1.5 + RIGHT * 3
        offsets = [(-0.8, 0.5), (0.6, -0.3), (-0.2, -0.6), (0.9, 0.4),
                   (-0.5, -0.1), (0.3, 0.7), (-0.7, -0.5), (0.1, 0.2)]
        for dx, dy in offsets:
            d = Dot(center_high + RIGHT * dx + UP * dy,
                     radius=0.06, color="#E74C3C")
            high_dots.add(d)

        self.play(FadeIn(low_var_label), FadeIn(high_var_label))
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in low_dots], lag_ratio=0.08),
                  LaggedStart(*[FadeIn(d, scale=0.5) for d in high_dots], lag_ratio=0.08))
        self.wait(0.5)

        rl_note = Text("In RL: high variance gradients = unstable training",
                        font_size=20, color="#f0883e", font=_FONT)
        rl_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(rl_note))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # 3. IMPORTANCE SAMPLING
        # ════════════════════════════════════════
        sec3 = Text("3. Importance Sampling", font_size=32, weight=BOLD, color="#58a6ff", font=_FONT)
        sec3.to_edge(UP, buff=0.5)
        self.play(Write(sec3), run_time=0.7)

        problem = Text("Problem: we have samples from p(x),", font_size=22, color=WHITE, font=_FONT)
        problem2 = Text("but want to evaluate E under q(x)", font_size=22, color=WHITE, font=_FONT)
        problem.next_to(sec3, DOWN, buff=0.5)
        problem2.next_to(problem, DOWN, buff=0.2)
        self.play(Write(problem), run_time=0.6)
        self.play(Write(problem2), run_time=0.6)
        self.wait(0.5)

        # The trick — proper LaTeX
        trick = MathTex(
            r"\mathbb{E}_{q}[f(x)] = \mathbb{E}_{p}\!\left[\frac{q(x)}{p(x)} \, f(x)\right]",
            font_size=36, color="#3fb950",
        )
        trick.next_to(problem2, DOWN, buff=0.5)
        self.play(Write(trick), run_time=0.8)

        ratio_label = MathTex(
            r"\frac{q(x)}{p(x)} = \text{importance weight}",
            font_size=32, color="#d2a8ff",
        )
        ratio_label.next_to(trick, DOWN, buff=0.3)
        self.play(FadeIn(ratio_label))
        self.wait(0.5)

        # PPO connection
        ppo_note = Text("In PPO: data from old policy, evaluate new policy",
                         font_size=20, color="#f0883e", font=_FONT)
        ppo_note.next_to(ratio_label, DOWN, buff=0.5)
        self.play(FadeIn(ppo_note), run_time=0.5)

        ppo_ratio = MathTex(
            r"r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}",
            font_size=34, color="#f0883e",
        )
        ppo_ratio.next_to(ppo_note, DOWN, buff=0.2)
        self.play(Write(ppo_ratio), run_time=0.7)

        warn = Text("Caveat: if p and q are very different, weights blow up",
                     font_size=18, color="#E74C3C", font=_FONT)
        warn.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(warn))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # 4. KL DIVERGENCE
        # ════════════════════════════════════════
        sec4 = Text("4. KL Divergence", font_size=32, weight=BOLD, color="#58a6ff", font=_FONT)
        sec4.to_edge(UP, buff=0.5)
        self.play(Write(sec4), run_time=0.7)

        kl_def = MathTex(
            r"D_\text{KL}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)}",
            font_size=36, color=WHITE,
        )
        kl_def.next_to(sec4, DOWN, buff=0.5)
        self.play(Write(kl_def), run_time=0.8)

        meaning_kl = Text("Measures how different two distributions are",
                           font_size=22, color=_SUB, font=_FONT)
        meaning_kl.next_to(kl_def, DOWN, buff=0.3)
        self.play(FadeIn(meaning_kl))
        self.wait(0.5)

        # Properties — LaTeX for math parts
        props = VGroup(
            MathTex(r"D_\text{KL} \geq 0 \;\;\text{always}", font_size=30, color=WHITE),
            MathTex(r"D_\text{KL} = 0 \;\;\text{only when } p = q", font_size=30, color=WHITE),
            MathTex(r"D_\text{KL}(p\|q) \neq D_\text{KL}(q\|p) \;\;\text{(not symmetric)}",
                    font_size=30, color=WHITE),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        props.next_to(meaning_kl, DOWN, buff=0.4)
        for p in props:
            self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.4)
        self.wait(0.5)

        # Visual: two distributions
        dist_label = Text("Small KL: distributions are similar",
                           font_size=20, color="#3fb950", font=_FONT)
        dist_label2 = Text("Large KL: distributions are very different",
                            font_size=20, color="#E74C3C", font=_FONT)
        g = VGroup(dist_label, dist_label2).arrange(DOWN, buff=0.2)
        g.next_to(props, DOWN, buff=0.5)
        self.play(FadeIn(dist_label), run_time=0.4)
        self.play(FadeIn(dist_label2), run_time=0.4)

        ppo_kl = Text("In PPO: constrain KL to keep policy updates safe",
                        font_size=20, color="#f0883e", font=_FONT)
        ppo_kl.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(ppo_kl))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── Summary ──
        summary_title = Text("Summary", font_size=32, weight=BOLD, color="#58a6ff", font=_FONT)
        summary_title.to_edge(UP, buff=0.5)
        self.play(Write(summary_title), run_time=0.5)

        items = VGroup(
            Text("1. Expectation  —  estimate via sampling", font_size=22, color=WHITE, font=_FONT),
            Text("2. Variance  —  lower is better for gradients", font_size=22, color=WHITE, font=_FONT),
            Text("3. Importance Sampling  —  reuse old data for new policy",
                 font_size=22, color=WHITE, font=_FONT),
            Text("4. KL Divergence  —  measure policy change", font_size=22, color=WHITE, font=_FONT),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        items.next_to(summary_title, DOWN, buff=0.6)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.5)
            self.wait(0.3)

        box = SurroundingRectangle(items, color="#f0883e", buff=0.3, corner_radius=0.1)
        self.play(Create(box), run_time=0.5)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
