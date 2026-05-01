# 10 — Alignment Roadmap Timeline
#
# Tool: Manim
# Output: MP4
#
# Animated timeline placing this paper in the RLHF → PPO → Summarize → InstructGPT
# → DPO lineage. Shows what each paper contributed to the alignment stack.
#
# Run:
#   manim -qm --media_dir ../output/animations 10_alignment_timeline.py AlignmentTimeline

from manim import *
import atexit
import shutil
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "2009.01325"

_FONT = "Latin Modern Roman"
_SUB = "#c9d1d9"  # high-contrast secondary text


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
        title = Text("The Alignment Roadmap", font_size=36,
                      weight=BOLD, color=WHITE, font=_FONT)
        sub = Text("From reward learning to direct preference optimization",
                    font_size=22, color=_SUB, font=_FONT)
        header = VGroup(title, sub).arrange(DOWN, buff=0.25).to_edge(UP, buff=0.5)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2), run_time=1.0)
        self.wait(0.5)

        # ── Timeline spine ──
        line_start = LEFT * 6 + DOWN * 0.3
        line_end = RIGHT * 6 + DOWN * 0.3
        spine = Line(line_start, line_end, color="#30363d", stroke_width=2)
        self.play(Create(spine), run_time=0.6)

        # ── Paper nodes ──
        papers = [
            {"year": "2017", "name": "RLHF",
             "contrib": "Learn rewards from\nhuman preferences",
             "color": "#58a6ff", "x": -4.5},
            {"year": "2017", "name": "PPO",
             "contrib": "Stable policy gradient\nwith clipping",
             "color": "#d2a8ff", "x": -1.8},
            {"year": "2020", "name": "Summarize\nfrom HF",
             "contrib": "RLHF scales to\nlanguage tasks",
             "color": "#f0883e", "x": 0.9},
            {"year": "2022", "name": "InstructGPT",
             "contrib": "RLHF at scale\nfor instruction following",
             "color": "#3fb950", "x": 3.6},
            {"year": "2023", "name": "DPO",
             "contrib": "No reward model\ndirect optimization",
             "color": "#E74C3C", "x": 5.8},
        ]

        for i, p in enumerate(papers):
            # Dot on timeline
            dot = Dot(point=[p["x"], -0.3, 0], radius=0.1, color=p["color"])

            # Year label below
            year = Text(p["year"], font_size=18, color=_SUB, font=_FONT)
            year.next_to(dot, DOWN, buff=0.2)

            # Name above
            name = Text(p["name"], font_size=20, color=p["color"],
                        weight=BOLD, font=_FONT)
            name.next_to(dot, UP, buff=0.25)

            # Contribution
            contrib = Text(p["contrib"], font_size=16, color=_SUB, font=_FONT)
            contrib.next_to(name, UP, buff=0.2)

            node = VGroup(dot, year, name, contrib)

            if i == 2:  # Highlight current paper
                highlight = SurroundingRectangle(
                    VGroup(name, contrib), color=p["color"],
                    buff=0.15, stroke_width=2, corner_radius=0.08
                )
                current_label = Text("← this paper", font_size=16,
                                      color=p["color"], font=_FONT)
                current_label.next_to(highlight, RIGHT, buff=0.2)
                self.play(FadeIn(node), Create(highlight), FadeIn(current_label),
                          run_time=0.8)
            else:
                self.play(FadeIn(node), run_time=0.6)
            self.wait(0.3)

        self.wait(1.0)

        # ── Connection arrows ──
        arrow_note = Text(
            "Each paper builds on the previous — same team, same pipeline, increasing scale",
            font_size=18, color=_SUB, font=_FONT
        )
        arrow_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(arrow_note), run_time=0.6)
        self.wait(2.5)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
