"""
Visualization 3: Preference Elicitation Flow
Detailed Manim animation of how trajectory segments are compared.

Shows:
1. Agent generates trajectory buffer
2. Two segments selected (with ensemble disagreement for query selection)
3. Shown side-by-side to human
4. Human picks preference -> stored as (σ¹, σ², μ)
5. Feeds into reward model training

Run: manim -ql --media_dir ../output/animations 03_preference_elicitation.py PreferenceElicitation
"""

from manim import *
import numpy as np


class PreferenceElicitation(Scene):
    def construct(self):
        # Title
        title = Text("Preference Elicitation Process", font_size=38, weight=BOLD)
        subtitle = Text("How human feedback is collected", font_size=22, color=GREY)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # === Step 1: Trajectory Buffer ===
        step1_title = Text("Step 1: Agent generates trajectories", font_size=22, color=GREEN, weight=BOLD)
        step1_title.to_edge(UP, buff=0.4)
        self.play(Write(step1_title))

        # Draw a trajectory as a sequence of dots
        def make_trajectory(start_pos, color, seed=42):
            rng = np.random.RandomState(seed)
            points = [start_pos]
            for _ in range(12):
                dx = 0.35 + rng.uniform(-0.05, 0.05)
                dy = rng.uniform(-0.25, 0.25)
                points.append(points[-1] + np.array([dx, dy, 0]))
            dots = VGroup(*[Dot(p, radius=0.06, color=color) for p in points])
            lines = VGroup(*[Line(points[i], points[i + 1], color=color, stroke_width=2)
                             for i in range(len(points) - 1)])
            return VGroup(lines, dots), points

        traj1, pts1 = make_trajectory(LEFT * 5.5 + UP * 1, BLUE, seed=42)
        traj2, pts2 = make_trajectory(LEFT * 5.5 + DOWN * 0.5, TEAL, seed=99)
        traj3, pts3 = make_trajectory(LEFT * 5.5 + DOWN * 2, BLUE_C, seed=17)

        buffer_label = Text("Trajectory Buffer", font_size=16, color=GREY)
        buffer_label.move_to(LEFT * 5.5 + DOWN * 3)

        self.play(
            *[Create(t, run_time=1.5) for t in [traj1, traj2, traj3]],
            FadeIn(buffer_label)
        )
        self.wait(0.5)

        # === Step 2: Select segments ===
        step2_title = Text("Step 2: Select two segments to compare", font_size=22, color=PURPLE, weight=BOLD)
        step2_title.to_edge(UP, buff=0.4)
        self.play(FadeOut(step1_title), Write(step2_title))

        # Highlight segments from trajectories
        seg1_rect = SurroundingRectangle(
            VGroup(*[Dot(pts1[i], radius=0.06) for i in range(3, 8)]),
            color=ORANGE, buff=0.15, corner_radius=0.1
        )
        seg1_label = Text("σ¹", font_size=20, color=ORANGE, weight=BOLD)
        seg1_label.next_to(seg1_rect, UP, buff=0.1)

        seg2_rect = SurroundingRectangle(
            VGroup(*[Dot(pts2[i], radius=0.06) for i in range(5, 10)]),
            color=YELLOW, buff=0.15, corner_radius=0.1
        )
        seg2_label = Text("σ²", font_size=20, color=YELLOW, weight=BOLD)
        seg2_label.next_to(seg2_rect, DOWN, buff=0.1)

        self.play(
            Create(seg1_rect), FadeIn(seg1_label),
            Create(seg2_rect), FadeIn(seg2_label),
            run_time=1
        )

        # Query selection note
        query_note = Text(
            "Selected via ensemble disagreement\n(highest variance across reward predictors)",
            font_size=14, color=GREY
        )
        query_note.move_to(RIGHT * 3 + UP * 1.5)
        self.play(FadeIn(query_note))
        self.wait(1)

        # === Step 3: Show to human ===
        step3_title = Text("Step 3: Human compares the two clips", font_size=22, color=MAROON, weight=BOLD)
        step3_title.to_edge(UP, buff=0.4)

        # Fade out trajectories, bring segments center
        self.play(
            FadeOut(step2_title), Write(step3_title),
            FadeOut(traj1), FadeOut(traj2), FadeOut(traj3),
            FadeOut(buffer_label), FadeOut(query_note),
            FadeOut(seg2_rect), FadeOut(seg1_rect),
        )

        # Two video-like boxes
        clip1_box = Rectangle(width=3, height=2, color=ORANGE, stroke_width=3)
        clip1_box.set_fill(ORANGE, opacity=0.08)
        clip1_box.move_to(LEFT * 2.5 + DOWN * 0.3)
        clip1_title = Text("Clip σ¹", font_size=20, color=ORANGE, weight=BOLD)
        clip1_title.next_to(clip1_box, UP, buff=0.1)
        # Simulate content: a simple sine wave "behavior"
        t_vals = np.linspace(0, 2 * np.pi, 50)
        wave1 = VMobject(color=ORANGE, stroke_width=2)
        wave1.set_points_smoothly([
            clip1_box.get_center() + np.array([0.8 * (t / (2 * np.pi) - 0.5) * 3, 0.4 * np.sin(t + 0.5), 0])
            for t in t_vals
        ])

        clip2_box = Rectangle(width=3, height=2, color=YELLOW, stroke_width=3)
        clip2_box.set_fill(YELLOW, opacity=0.08)
        clip2_box.move_to(RIGHT * 2.5 + DOWN * 0.3)
        clip2_title = Text("Clip σ²", font_size=20, color=YELLOW, weight=BOLD)
        clip2_title.next_to(clip2_box, UP, buff=0.1)
        wave2 = VMobject(color=YELLOW, stroke_width=2)
        wave2.set_points_smoothly([
            clip2_box.get_center() + np.array([0.8 * (t / (2 * np.pi) - 0.5) * 3, 0.3 * np.sin(2 * t), 0])
            for t in t_vals
        ])

        vs_text = Text("VS", font_size=28, color=RED, weight=BOLD)
        vs_text.move_to(DOWN * 0.3)

        self.play(
            FadeIn(clip1_box), FadeIn(clip1_title), Create(wave1),
            FadeIn(clip2_box), FadeIn(clip2_title), Create(wave2),
            FadeIn(vs_text),
            FadeOut(seg1_label), FadeOut(seg2_label),
            run_time=1.2
        )
        self.wait(0.5)

        # Duration note
        duration = Text("1-2 seconds each • Human responds in 3-5 seconds", font_size=14, color=GREY)
        duration.move_to(DOWN * 2)
        self.play(FadeIn(duration))
        self.wait(0.5)

        # === Step 4: Human picks ===
        step4_title = Text("Step 4: Human indicates preference", font_size=22, color=GOLD, weight=BOLD)
        step4_title.to_edge(UP, buff=0.4)
        self.play(FadeOut(step3_title), Write(step4_title))

        # Three options
        opt_left = Text("σ¹ better", font_size=16, color=ORANGE)
        opt_equal = Text("Equal", font_size=16, color=WHITE)
        opt_right = Text("σ² better", font_size=16, color=YELLOW)
        opt_skip = Text("Can't tell", font_size=14, color=GREY)

        options = VGroup(opt_left, opt_equal, opt_right, opt_skip).arrange(RIGHT, buff=0.8)
        options.move_to(DOWN * 2.8)

        self.play(FadeOut(duration), FadeIn(options))
        self.wait(0.3)

        # Human picks σ¹
        pick_box = SurroundingRectangle(opt_left, color=GREEN, buff=0.1, corner_radius=0.1)
        self.play(Create(pick_box), run_time=0.5)

        # Highlight clip 1
        self.play(
            clip1_box.animate.set_stroke(color=GREEN, width=5),
            clip2_box.animate.set_stroke(color=GREY, width=1),
            run_time=0.5
        )
        self.wait(0.5)

        # === Step 5: Store ===
        step5_title = Text("Step 5: Store in database D", font_size=22, color=TEAL, weight=BOLD)
        step5_title.to_edge(UP, buff=0.4)
        self.play(FadeOut(step4_title), Write(step5_title))

        # Triple display
        triple = Text(
            "(σ¹, σ², μ) where μ = (1, 0)",
            font_size=24, color=TEAL, weight=BOLD
        )
        triple.move_to(DOWN * 3.2)

        # Explanation
        expl = Text(
            "μ = (1, 0): all mass on σ¹  |  μ = (0.5, 0.5): equal  |  incomparable: discarded",
            font_size=13, color=GREY
        )
        expl.move_to(DOWN * 3.8)

        self.play(
            FadeOut(options), FadeOut(pick_box),
            Write(triple), FadeIn(expl),
            run_time=1
        )
        self.wait(2)

        # Final summary
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        summary = VGroup(
            Text("Preference Elicitation Summary", font_size=28, weight=BOLD, color=WHITE),
            Text("", font_size=10),
            Text("1. Agent explores → trajectory buffer", font_size=18, color=GREEN),
            Text("2. Select pairs via ensemble disagreement", font_size=18, color=PURPLE),
            Text("3. Show 1-2 sec clips to human side-by-side", font_size=18, color=MAROON),
            Text("4. Human picks: σ¹ better / σ² better / equal / skip", font_size=18, color=GOLD),
            Text("5. Store (σ¹, σ², μ) in database D", font_size=18, color=TEAL),
            Text("", font_size=10),
            Text("Only ~700 comparisons needed for MuJoCo tasks!", font_size=18, color=ORANGE, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        summary.move_to(ORIGIN)

        for item in summary:
            self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.3)
        self.wait(3)
