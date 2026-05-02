# 10 — Alignment Timeline (Fresh Standalone)
#
# Tool: Manim
# Output: MP4
#
# Timeline centered on InstructGPT showing the lineage:
# RLHF (2017) → PPO (2017) → Fine-Tuning LMs from HP (2019) →
# Summarization from HF (2020) → InstructGPT (2022) → ChatGPT (2022) → DPO (2023)
#
# Run:
#   manim -qm --media_dir ../output/animations 10_alignment_timeline.py AlignmentTimeline

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
    src = _PAPER_DIR / "output/animations/videos/10_alignment_timeline/720p30/AlignmentTimeline.mp4"
    dst = _DOCS_DIR / "AlignmentTimeline.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


class AlignmentTimeline(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("The Road to InstructGPT", font_size=36,
                      weight=BOLD, color=WHITE, font=_FONT)
        sub = Text("Alignment research timeline: from RLHF to ChatGPT",
                    font_size=22, color=_SUB, font=_FONT)
        header = VGroup(title, sub).arrange(DOWN, buff=0.25).to_edge(UP, buff=0.5)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2), run_time=1.0)
        self.wait(0.5)

        # ── Timeline data ──
        events = [
            ("2017", "RLHF", "Christiano et al.\nHuman preferences\nfor deep RL", "#58a6ff"),
            ("2017", "PPO", "Schulman et al.\nClipped surrogate\nobjective", "#58a6ff"),
            ("2019", "Fine-Tuning\nfrom HP", "Ziegler et al.\nRLHF for text\n(stylistic tasks)", "#d2a8ff"),
            ("2020", "Summarize\nfrom HF", "Stiennon et al.\nRLHF for TL;DR\n(SFT→RM→PPO)", "#d2a8ff"),
            ("2022", "InstructGPT", "Ouyang et al.\nRLHF at scale\n(PPO-ptx, K-way)", "#3fb950"),
            ("2022", "ChatGPT", "OpenAI\nInstructGPT applied\nto dialogue", "#f0883e"),
            ("2023", "DPO", "Rafailov et al.\nDirect preference\noptimization", "#f85149"),
        ]

        # ── Build the horizontal timeline ──
        line = Line(LEFT * 5.5, RIGHT * 5.5, color=_SUB, stroke_width=2)
        line.shift(DOWN * 0.2)
        self.play(Create(line), run_time=0.8)

        n = len(events)
        x_positions = [line.get_left()[0] + i * (11.0 / (n - 1)) for i in range(n)]

        dots = VGroup()
        year_labels = VGroup()
        name_labels = VGroup()
        desc_labels = VGroup()

        for i, (year, name, desc, color) in enumerate(events):
            x = x_positions[i]
            pos = np.array([x, line.get_center()[1], 0])

            # Dot on timeline
            dot = Dot(pos, radius=0.1, color=color, fill_opacity=1.0)
            dots.add(dot)

            # Year below
            yr = Text(year, font_size=16, color=_SUB, font=_FONT)
            yr.next_to(dot, DOWN, buff=0.2)
            year_labels.add(yr)

            # Name above (alternating up/down for readability)
            goes_up = (i % 2 == 0)
            nm = Text(name, font_size=18, color=color, weight=BOLD, font=_FONT)

            if goes_up:
                nm.next_to(dot, UP, buff=0.4)
            else:
                nm.next_to(yr, DOWN, buff=0.3)
            name_labels.add(nm)

            # Description
            ds = Text(desc, font_size=14, color=_SUB, font=_FONT, line_spacing=1.0)
            if goes_up:
                ds.next_to(nm, UP, buff=0.15)
            else:
                ds.next_to(nm, DOWN, buff=0.15)
            desc_labels.add(ds)

        # Animate events appearing left to right
        for i in range(n):
            anims = [
                FadeIn(dots[i], scale=1.5),
                FadeIn(year_labels[i]),
                FadeIn(name_labels[i], shift=UP * 0.15 if i % 2 == 0 else DOWN * 0.15),
            ]
            self.play(*anims, run_time=0.6)
            self.play(FadeIn(desc_labels[i], shift=UP * 0.1), run_time=0.4)

            # Highlight InstructGPT specially
            if events[i][1] == "InstructGPT":
                highlight = SurroundingRectangle(
                    VGroup(name_labels[i], desc_labels[i]),
                    color="#3fb950", buff=0.15, stroke_width=2
                )
                self.play(Create(highlight), run_time=0.5)
                self.wait(0.5)

        self.wait(1.0)

        # ── Arrow showing lineage ──
        lineage_note = Text(
            "Direct lineage: each paper builds on the previous",
            font_size=20, color="#f0883e", font=_FONT
        )
        lineage_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(lineage_note, shift=UP * 0.2), run_time=0.6)
        self.wait(2.0)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
