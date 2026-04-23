"""
Visualization 2: RL vs RLHF Pipeline Comparison
Side-by-side Manim animation comparing:
  Left:  Traditional RL (Agent -> Environment -> Reward -> Agent)
  Right: RLHF (Agent -> Environment -> Human Prefs -> Reward Model -> Agent)

Run: manim -ql --media_dir ../output/animations 02_rl_vs_rlhf_pipeline.py RLvsRLHF
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
    src = _PAPER_DIR / "output/animations/videos/02_rl_vs_rlhf_pipeline/480p15/RLvsRLHF.mp4"
    dst = _DOCS_DIR / "RLvsRLHF.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst}")

atexit.register(_copy_to_docs)


class RLvsRLHF(Scene):
    def construct(self):
        # Title
        title = Text("Traditional RL vs RLHF", font_size=42, weight=BOLD)
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        self.play(title.animate.scale(0.6).to_edge(UP, buff=0.3))

        # === LEFT: Traditional RL ===
        left_title = Text("Traditional RL", font_size=24, weight=BOLD, color=BLUE)
        left_title.move_to(LEFT * 3.5 + UP * 2.2)

        # Boxes
        def make_box(label_text, color, pos, sublabel=None):
            box = RoundedRectangle(
                corner_radius=0.15, width=2.2, height=0.9, color=color
            ).set_fill(color, opacity=0.15).move_to(pos)
            label = Text(label_text, font_size=16, weight=BOLD).move_to(box.get_center())
            if sublabel:
                sub = Text(sublabel, font_size=11, color=GREY).move_to(box.get_center() + DOWN * 0.2)
                label.move_to(box.get_center() + UP * 0.1)
                return VGroup(box, label, sub)
            return VGroup(box, label)

        # Left column
        l_agent = make_box("Agent (Policy)", GREEN, LEFT * 3.5 + UP * 1)
        l_env = make_box("Environment", BLUE, LEFT * 3.5 + DOWN * 0.3)
        l_reward = make_box("Reward r(s,a)", ORANGE, LEFT * 3.5 + DOWN * 1.6)

        # Right column: RLHF
        right_title = Text("RLHF (This Paper)", font_size=24, weight=BOLD, color=PURPLE)
        right_title.move_to(RIGHT * 3.5 + UP * 2.2)

        r_agent = make_box("Agent (Policy)", GREEN, RIGHT * 3.5 + UP * 1)
        r_env = make_box("Environment", BLUE, RIGHT * 3.5 + DOWN * 0.3)
        r_human = make_box("Human", PURPLE, RIGHT * 2 + DOWN * 1.6, "preferences")
        r_reward = make_box("Reward Model", ORANGE, RIGHT * 5 + DOWN * 1.6, "r̂(o, a)")

        # Divider
        divider = DashedLine(UP * 2.5, DOWN * 3, color=GREY, dash_length=0.15)

        # Show structure
        self.play(FadeIn(left_title), FadeIn(right_title), Create(divider))
        self.play(
            *[FadeIn(g, scale=0.8) for g in [l_agent, l_env, l_reward,
                                               r_agent, r_env, r_human, r_reward]],
            run_time=1.5
        )
        self.wait(0.5)

        # === LEFT arrows: simple loop ===
        def make_arrow(start_mob, end_mob, direction='down', color=WHITE, label_text=None, label_side=LEFT):
            start = start_mob[0].get_bottom() if direction == 'down' else start_mob[0].get_top()
            end = end_mob[0].get_top() if direction == 'down' else end_mob[0].get_bottom()
            arr = Arrow(start, end, buff=0.08, color=color, stroke_width=2.5, max_tip_length_to_length_ratio=0.15)
            if label_text:
                label = Text(label_text, font_size=11, color=color)
                label.next_to(arr, label_side, buff=0.05)
                return arr, label
            return arr, None

        # Left: Agent -> Env
        la1, ll1 = make_arrow(l_agent, l_env, 'down', GREEN, "action", LEFT)
        # Left: Env -> Reward (built-in)
        la2, ll2 = make_arrow(l_env, l_reward, 'down', BLUE, "state", LEFT)
        # Left: Reward -> Agent (curved back)
        la3 = CurvedArrow(
            l_reward[0].get_left() + DOWN * 0.1,
            l_agent[0].get_left() + DOWN * 0.1,
            angle=-TAU / 4, color=ORANGE, stroke_width=2.5
        )
        ll3 = Text("reward r", font_size=11, color=ORANGE)
        ll3.next_to(la3, LEFT, buff=0.05)

        step_l = Text("Simple closed loop", font_size=16, color=BLUE)
        step_l.move_to(LEFT * 3.5 + DOWN * 2.8)

        self.play(
            GrowArrow(la1), FadeIn(ll1),
            GrowArrow(la2), FadeIn(ll2),
            Create(la3), FadeIn(ll3),
            FadeIn(step_l),
            run_time=1.5
        )
        self.wait(1)

        # === RIGHT arrows: RLHF loop ===
        # Agent -> Env
        ra1, rl1 = make_arrow(r_agent, r_env, 'down', GREEN, "action", RIGHT)

        # Env -> Human (segments)
        ra2 = Arrow(
            r_env[0].get_bottom() + LEFT * 0.3,
            r_human[0].get_top(),
            buff=0.08, color=PURPLE, stroke_width=2.5, max_tip_length_to_length_ratio=0.15
        )
        rl2 = Text("segments", font_size=11, color=PURPLE)
        rl2.next_to(ra2, LEFT, buff=0.05)

        # Human -> Reward Model
        ra3 = Arrow(
            r_human[0].get_right(),
            r_reward[0].get_left(),
            buff=0.08, color=YELLOW, stroke_width=2.5, max_tip_length_to_length_ratio=0.15
        )
        rl3 = Text("(σ¹,σ²,μ)", font_size=11, color=YELLOW)
        rl3.next_to(ra3, DOWN, buff=0.05)

        # Reward Model -> Agent (curved back)
        ra4 = CurvedArrow(
            r_reward[0].get_right() + UP * 0.1,
            r_agent[0].get_right() + DOWN * 0.1,
            angle=-TAU / 4, color=ORANGE, stroke_width=2.5
        )
        rl4 = Text("predicted\nreward r̂", font_size=11, color=ORANGE)
        rl4.next_to(ra4, RIGHT, buff=0.05)

        step_r = Text("Human-in-the-loop", font_size=16, color=PURPLE)
        step_r.move_to(RIGHT * 3.5 + DOWN * 2.8)

        self.play(
            GrowArrow(ra1), FadeIn(rl1),
            run_time=0.5
        )
        self.play(
            GrowArrow(ra2), FadeIn(rl2),
            run_time=0.5
        )
        self.play(
            GrowArrow(ra3), FadeIn(rl3),
            run_time=0.5
        )
        self.play(
            Create(ra4), FadeIn(rl4),
            FadeIn(step_r),
            run_time=0.7
        )
        self.wait(1)

        # === Key differences ===
        diff_title = Text("Key Differences", font_size=22, weight=BOLD, color=YELLOW)
        diff_title.move_to(DOWN * 3.2)

        diffs = VGroup(
            Text("• No hand-designed reward function needed", font_size=15, color=WHITE),
            Text("• Human provides ~1% of feedback (very efficient)", font_size=15, color=WHITE),
            Text("• Reward model learned via Bradley-Terry + cross-entropy", font_size=15, color=WHITE),
            Text("• All 3 processes run asynchronously", font_size=15, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        diffs.move_to(DOWN * 3.2)

        # Highlight the replaced component
        cross = Cross(l_reward[0], stroke_color=RED, stroke_width=4)
        new_label = Text("REPLACED", font_size=14, color=RED, weight=BOLD)
        new_label.next_to(l_reward, DOWN, buff=0.1)

        self.play(Create(cross), FadeIn(new_label), run_time=0.7)
        self.wait(0.5)

        # Scroll down to show diffs
        all_upper = VGroup(
            left_title, right_title, divider,
            l_agent, l_env, l_reward,
            r_agent, r_env, r_human, r_reward,
            la1, ll1, la2, ll2, la3, ll3,
            ra1, rl1, ra2, rl2, ra3, rl3, ra4, rl4,
            step_l, step_r, cross, new_label
        )
        self.play(all_upper.animate.shift(UP * 1.5), run_time=0.8)
        self.play(FadeIn(diff_title, shift=UP * 0.3))
        for d in diffs:
            self.play(FadeIn(d, shift=RIGHT * 0.3), run_time=0.4)

        self.wait(3)
