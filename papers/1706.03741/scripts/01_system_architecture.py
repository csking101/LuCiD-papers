"""
Visualization 1: System Architecture
Animated Manim diagram of the three asynchronous processes from Figure 1.

Shows:
1. Policy interacts with environment -> trajectories
2. Trajectory segments selected -> sent to human for comparison
3. Human comparisons -> train reward model -> reward signal back to policy

Run: manim -ql --media_dir ../output/animations 01_system_architecture.py SystemArchitecture
Output: ../output/animations/videos/01_system_architecture/480p15/SystemArchitecture.mp4
"""

from manim import *
import atexit
import shutil
from pathlib import Path

# Post-render: copy MP4 to docs/ for GitHub Pages serving
_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "1706.03741"

def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/01_system_architecture/480p15/SystemArchitecture.mp4"
    dst = _DOCS_DIR / "SystemArchitecture.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst}")

atexit.register(_copy_to_docs)


class SystemArchitecture(Scene):
    def construct(self):
        # Title
        title = Text("RLHF System Architecture", font_size=40, weight=BOLD)
        subtitle = Text(
            "Christiano et al., 2017 — Three Asynchronous Processes",
            font_size=22, color=GREY
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.7)
        self.wait(0.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # === Main diagram ===

        # Boxes
        env_box = RoundedRectangle(
            corner_radius=0.2, width=2.8, height=1.2, color=BLUE
        ).set_fill(BLUE, opacity=0.15)
        env_label = Text("Environment", font_size=20, weight=BOLD)
        env_group = VGroup(env_box, env_label).move_to(LEFT * 4.5 + UP * 1.5)
        env_label.move_to(env_box.get_center())

        policy_box = RoundedRectangle(
            corner_radius=0.2, width=2.8, height=1.2, color=GREEN
        ).set_fill(GREEN, opacity=0.15)
        policy_label = Text("Policy π", font_size=20, weight=BOLD)
        policy_sub = Text("(A2C / TRPO)", font_size=14, color=GREY)
        policy_group = VGroup(policy_box, policy_label, policy_sub).move_to(LEFT * 4.5 + DOWN * 1.5)
        policy_label.move_to(policy_box.get_center() + UP * 0.15)
        policy_sub.move_to(policy_box.get_center() + DOWN * 0.25)

        human_box = RoundedRectangle(
            corner_radius=0.2, width=2.8, height=1.2, color=PURPLE
        ).set_fill(PURPLE, opacity=0.15)
        human_label = Text("Human", font_size=20, weight=BOLD)
        human_sub = Text("(Comparisons)", font_size=14, color=GREY)
        human_group = VGroup(human_box, human_label, human_sub).move_to(RIGHT * 0 + UP * 1.5)
        human_label.move_to(human_box.get_center() + UP * 0.15)
        human_sub.move_to(human_box.get_center() + DOWN * 0.25)

        reward_box = RoundedRectangle(
            corner_radius=0.2, width=2.8, height=1.2, color=ORANGE
        ).set_fill(ORANGE, opacity=0.15)
        reward_label = Text("Reward Model", font_size=20, weight=BOLD)
        reward_sub = Text("r̂(o, a)", font_size=16, color=GREY)
        reward_group = VGroup(reward_box, reward_label, reward_sub).move_to(RIGHT * 4.5 + DOWN * 1.5)
        reward_label.move_to(reward_box.get_center() + UP * 0.15)
        reward_sub.move_to(reward_box.get_center() + DOWN * 0.25)

        # Database
        db_box = RoundedRectangle(
            corner_radius=0.1, width=2.0, height=0.8, color=YELLOW
        ).set_fill(YELLOW, opacity=0.15)
        db_label = Text("Database D", font_size=16, weight=BOLD)
        db_group = VGroup(db_box, db_label).move_to(RIGHT * 4.5 + UP * 1.5)
        db_label.move_to(db_box.get_center())

        # Step 1: Show all boxes
        self.play(
            *[FadeIn(g, scale=0.8) for g in
              [env_group, policy_group, human_group, reward_group, db_group]],
            run_time=1.5
        )
        self.wait(0.3)

        # === Arrows with labels ===

        # Arrow 1: Policy <-> Environment (bidirectional)
        arr_pe = Arrow(
            policy_box.get_top(), env_box.get_bottom(),
            buff=0.1, color=BLUE, stroke_width=3
        )
        arr_ep = Arrow(
            env_box.get_bottom() + RIGHT * 0.5, policy_box.get_top() + RIGHT * 0.5,
            buff=0.1, color=GREEN, stroke_width=3
        )
        arr_pe_shifted = Arrow(
            policy_box.get_top() + LEFT * 0.5, env_box.get_bottom() + LEFT * 0.5,
            buff=0.1, color=BLUE, stroke_width=3
        )
        label_pe = Text("actions", font_size=14, color=BLUE).next_to(arr_pe_shifted, LEFT, buff=0.1)
        label_ep = Text("observations\ntrajectories", font_size=12, color=GREEN).next_to(arr_ep, RIGHT, buff=0.1)

        # Step label
        step1 = Text("① Policy explores", font_size=18, color=GREEN, weight=BOLD)
        step1.move_to(UP * 3.2)

        self.play(Write(step1), run_time=0.5)
        self.play(
            GrowArrow(arr_pe_shifted), GrowArrow(arr_ep),
            FadeIn(label_pe), FadeIn(label_ep),
            run_time=1
        )
        self.wait(1)

        # Arrow 2: Environment -> Human (trajectory segments)
        arr_eh = Arrow(
            env_box.get_right(), human_box.get_left(),
            buff=0.1, color=PURPLE, stroke_width=3
        )
        label_eh = Text("trajectory\nsegments (σ¹, σ²)", font_size=12, color=PURPLE)
        label_eh.next_to(arr_eh, UP, buff=0.1)

        step2 = Text("② Select pairs, show to human", font_size=18, color=PURPLE, weight=BOLD)
        step2.move_to(UP * 3.2)

        self.play(FadeOut(step1), Write(step2), run_time=0.5)
        self.play(GrowArrow(arr_eh), FadeIn(label_eh), run_time=1)
        self.wait(1)

        # Arrow 3: Human -> Database
        arr_hd = Arrow(
            human_box.get_right(), db_box.get_left(),
            buff=0.1, color=YELLOW, stroke_width=3
        )
        label_hd = Text("(σ¹, σ², μ)", font_size=14, color=YELLOW)
        label_hd.next_to(arr_hd, UP, buff=0.1)

        step3 = Text("③ Human provides preference", font_size=18, color=YELLOW_D, weight=BOLD)
        step3.move_to(UP * 3.2)

        self.play(FadeOut(step2), Write(step3), run_time=0.5)
        self.play(GrowArrow(arr_hd), FadeIn(label_hd), run_time=1)
        self.wait(1)

        # Arrow 4: Database -> Reward Model
        arr_dr = Arrow(
            db_box.get_bottom(), reward_box.get_top(),
            buff=0.1, color=ORANGE, stroke_width=3
        )
        label_dr = Text("train via\ncross-entropy", font_size=12, color=ORANGE)
        label_dr.next_to(arr_dr, RIGHT, buff=0.1)

        step4 = Text("④ Train reward model on preferences", font_size=18, color=ORANGE, weight=BOLD)
        step4.move_to(UP * 3.2)

        self.play(FadeOut(step3), Write(step4), run_time=0.5)
        self.play(GrowArrow(arr_dr), FadeIn(label_dr), run_time=1)
        self.wait(1)

        # Arrow 5: Reward Model -> Policy
        arr_rp = Arrow(
            reward_box.get_left(), policy_box.get_right(),
            buff=0.1, color=RED, stroke_width=3
        )
        label_rp = Text("predicted\nreward r̂", font_size=12, color=RED)
        label_rp.next_to(arr_rp, DOWN, buff=0.1)

        step5 = Text("⑤ Policy optimizes predicted reward", font_size=18, color=RED, weight=BOLD)
        step5.move_to(UP * 3.2)

        self.play(FadeOut(step4), Write(step5), run_time=0.5)
        self.play(GrowArrow(arr_rp), FadeIn(label_rp), run_time=1)
        self.wait(1)

        # Final: Show all steps cycling
        final_label = Text(
            "All three processes run asynchronously",
            font_size=20, color=WHITE, weight=BOLD
        )
        final_label.move_to(UP * 3.2)
        self.play(FadeOut(step5), Write(final_label))

        # Pulse the cycle
        cycle_arrows = [arr_pe_shifted, arr_ep, arr_eh, arr_hd, arr_dr, arr_rp]
        for _ in range(2):
            for arrow in cycle_arrows:
                self.play(
                    arrow.animate.set_stroke(width=6),
                    run_time=0.15
                )
                self.play(
                    arrow.animate.set_stroke(width=3),
                    run_time=0.15
                )

        self.wait(2)
