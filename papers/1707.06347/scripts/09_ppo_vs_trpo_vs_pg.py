# 09 — PPO vs TRPO vs Vanilla PG
#
# Tool: Manim
# Output: MP4
#
# Side-by-side comparison animation of three agents learning:
# Vanilla PG is erratic (occasional collapses),
# TRPO is steady but slow (computing Fisher matrix),
# PPO is steady and fast (just clipping).
#
# Run:
#   manim -qm --media_dir ../output/animations 09_ppo_vs_trpo_vs_pg.py PPOvsTRPOvsPG

from manim import *
import atexit
import shutil
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "1707.06347"

_FONT = "Latin Modern Roman"
_SUB = "#c9d1d9"  # high-contrast secondary text


def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/09_ppo_vs_trpo_vs_pg/720p30/PPOvsTRPOvsPG.mp4"
    dst = _DOCS_DIR / "PPOvsTRPOvsPG.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


class PPOvsTRPOvsPG(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("PPO vs TRPO vs Vanilla Policy Gradient", font_size=34,
                      weight=BOLD, color=WHITE, font=_FONT)
        subtitle = Text("Three approaches to the same problem", font_size=22, color=_SUB, font=_FONT)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ════════════════════════════════════════
        # SETUP: Three parallel training curves
        # ════════════════════════════════════════
        header = Text("Training Performance Over Time", font_size=28, weight=BOLD,
                       color="#58a6ff", font=_FONT)
        header.to_edge(UP, buff=0.3)
        self.play(Write(header), run_time=0.5)

        # Create three side-by-side axes
        methods = [
            ("Vanilla PG", "#E74C3C"),
            ("TRPO", "#d2a8ff"),
            ("PPO", "#3fb950"),
        ]

        axes_group = VGroup()
        labels_group = VGroup()
        for i, (name, color) in enumerate(methods):
            ax = Axes(
                x_range=[0, 20, 5], y_range=[0, 100, 25],
                x_length=3.5, y_length=2.8,
                axis_config={"color": _SUB, "stroke_width": 1,
                             "include_numbers": False},
            )
            x_offset = (i - 1) * 4.2
            ax.move_to(RIGHT * x_offset + DOWN * 0.3)
            axes_group.add(ax)

            label = Text(name, font_size=20, weight=BOLD, color=color, font=_FONT)
            label.next_to(ax, UP, buff=0.2)
            labels_group.add(label)

        self.play(*[Create(ax) for ax in axes_group],
                  *[FadeIn(l) for l in labels_group])

        # Add axis labels
        for ax in axes_group:
            xl = Text("Updates", font_size=14, color=_SUB, font=_FONT)
            xl.next_to(ax.x_axis, DOWN, buff=0.15)
            yl = Text("Reward", font_size=14, color=_SUB, font=_FONT)
            yl.next_to(ax.y_axis, LEFT, buff=0.15)
            self.add(xl, yl)

        # ═══ Generate training curves ═══

        np.random.seed(42)
        n_steps = 20

        # Vanilla PG: noisy, occasional collapse
        vanilla_rewards = [10]
        for i in range(1, n_steps):
            if i == 8:  # collapse
                vanilla_rewards.append(5)
            elif i == 9:
                vanilla_rewards.append(2)
            elif i == 10:
                vanilla_rewards.append(8)
            else:
                next_r = vanilla_rewards[-1] + np.random.uniform(-8, 12)
                next_r = np.clip(next_r, 0, 100)
                vanilla_rewards.append(next_r)
        # Ensure it's volatile
        vanilla_rewards = [max(0, min(100, r + np.random.uniform(-5, 5))) for r in vanilla_rewards]

        # TRPO: steady but slow improvement
        trpo_rewards = [10]
        for i in range(1, n_steps):
            trpo_rewards.append(trpo_rewards[-1] + np.random.uniform(1, 4))
        trpo_rewards = [min(100, r) for r in trpo_rewards]

        # PPO: steady and faster improvement
        ppo_rewards = [10]
        for i in range(1, n_steps):
            ppo_rewards.append(ppo_rewards[-1] + np.random.uniform(2, 6))
        ppo_rewards = [min(100, r) for r in ppo_rewards]

        all_data = [vanilla_rewards, trpo_rewards, ppo_rewards]
        colors = ["#E74C3C", "#d2a8ff", "#3fb950"]

        # Animate curves simultaneously, point by point
        curves = [VGroup() for _ in range(3)]
        dots = [VGroup() for _ in range(3)]

        for step in range(n_steps):
            new_dots = []
            new_lines = []
            for j in range(3):
                ax = axes_group[j]
                x = step
                y = all_data[j][step]
                point = ax.c2p(x, y)
                dot = Dot(point, radius=0.04, color=colors[j])
                dots[j].add(dot)
                new_dots.append(dot)

                if step > 0:
                    prev_point = ax.c2p(step - 1, all_data[j][step - 1])
                    line = Line(prev_point, point, color=colors[j], stroke_width=2)
                    curves[j].add(line)
                    new_lines.append(line)

            anims = [FadeIn(d, scale=0.5) for d in new_dots]
            anims += [Create(l) for l in new_lines]

            if step == 0:
                self.play(*anims, run_time=0.15)
            else:
                self.play(*anims, run_time=0.12)

        self.wait(0.5)

        # ═══ Add annotations ═══
        vanilla_note = Text("Unstable, collapses", font_size=16, color="#E74C3C", font=_FONT)
        vanilla_note.next_to(axes_group[0], DOWN, buff=0.4)

        trpo_note = Text("Stable, but slow", font_size=16, color="#d2a8ff", font=_FONT)
        trpo_note.next_to(axes_group[1], DOWN, buff=0.4)

        ppo_note = Text("Stable AND fast", font_size=16, color="#3fb950", font=_FONT)
        ppo_note.next_to(axes_group[2], DOWN, buff=0.4)

        self.play(FadeIn(vanilla_note), FadeIn(trpo_note), FadeIn(ppo_note))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        # ════════════════════════════════════════
        # WHY PPO WINS
        # ════════════════════════════════════════
        why_label = Text("Why PPO Wins", font_size=30, weight=BOLD, color="#3fb950", font=_FONT)
        why_label.to_edge(UP, buff=0.4)
        self.play(Write(why_label), run_time=0.5)

        # Comparison table
        headers = VGroup(
            Text("Method", font_size=18, weight=BOLD, color=_SUB, font=_FONT),
            Text("Step Control", font_size=18, weight=BOLD, color=_SUB, font=_FONT),
            Text("Complexity", font_size=18, weight=BOLD, color=_SUB, font=_FONT),
            Text("Result", font_size=18, weight=BOLD, color=_SUB, font=_FONT),
        ).arrange(RIGHT, buff=1.0)
        headers.next_to(why_label, DOWN, buff=0.6)

        rows = [
            ("Vanilla PG", "None (fixed lr)", "Simple", "Unstable"),
            ("TRPO", "KL constraint", "2nd order", "Stable, slow"),
            ("PPO", "Clipping", "1st order", "Stable, fast"),
        ]
        row_colors = ["#E74C3C", "#d2a8ff", "#3fb950"]

        self.play(FadeIn(headers))

        table_rows = VGroup()
        for idx, (method, step_ctrl, complexity, result) in enumerate(rows):
            row = VGroup(
                Text(method, font_size=17, color=row_colors[idx], weight=BOLD, font=_FONT),
                Text(step_ctrl, font_size=17, color=WHITE, font=_FONT),
                Text(complexity, font_size=17, color=WHITE, font=_FONT),
                Text(result, font_size=17, color=row_colors[idx], font=_FONT),
            ).arrange(RIGHT, buff=1.0)
            table_rows.add(row)

        table_rows.arrange(DOWN, buff=0.35)
        table_rows.next_to(headers, DOWN, buff=0.4)

        # Align columns
        for row in table_rows:
            for i in range(4):
                row[i].move_to([headers[i].get_x(), row[i].get_y(), 0])

        for row in table_rows:
            self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.4)

        # Highlight PPO row
        ppo_row_highlight = SurroundingRectangle(table_rows[2], color="#3fb950",
                                                  buff=0.15, corner_radius=0.1)
        self.play(Create(ppo_row_highlight))
        self.wait(0.5)

        key_msg = Text("Same stability as TRPO, simplicity of vanilla PG",
                        font_size=20, weight=BOLD, color="#f0883e", font=_FONT)
        key_msg.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(key_msg))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])
