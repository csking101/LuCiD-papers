# 03 — K-way Ranking & Batching Strategy
#
# Tool: Manim
# Output: MP4
#
# Animate how K=4..9 outputs get ranked by labelers, producing C(K,2) comparison
# pairs. Show the naive approach (each pair as separate data point → overfits)
# vs batched approach (single batch element with all C(K,2) pairs → one forward
# pass per completion → less overfitting).
#
# Run:
#   manim -qm --media_dir ../output/animations 03_kway_ranking_batching.py KwayRankingBatching

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
    src = _PAPER_DIR / "output/animations/videos/03_kway_ranking_batching/720p30/KwayRankingBatching.mp4"
    dst = _DOCS_DIR / "KwayRankingBatching.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")


atexit.register(_copy_to_docs)


def _card(label, color, width=1.1, height=0.7):
    rect = RoundedRectangle(
        corner_radius=0.1, width=width, height=height,
        stroke_color=color, fill_color=color, fill_opacity=0.12,
        stroke_width=2
    )
    txt = Text(label, font_size=18, color=color, font=_FONT)
    txt.move_to(rect.get_center())
    return VGroup(rect, txt)


class KwayRankingBatching(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──
        title = Text("K-way Ranking & Batching", font_size=36,
                      weight=BOLD, color=WHITE, font=_FONT)
        sub = Text("Why batch all C(K,2) pairs from one prompt together",
                    font_size=22, color=_SUB, font=_FONT)
        header = VGroup(title, sub).arrange(DOWN, buff=0.25).to_edge(UP, buff=0.5)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2), run_time=1.0)
        self.wait(0.5)

        # ══════════════════════════════════════════════════════════════
        # SCENE 1: Show K outputs from one prompt
        # ══════════════════════════════════════════════════════════════
        s1_label = Text("Step 1: Generate K outputs per prompt",
                        font_size=26, color="#58a6ff", weight=BOLD, font=_FONT)
        s1_label.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(s1_label, shift=RIGHT * 0.3))

        prompt_box = _card("Prompt x", "#58a6ff", width=1.8, height=0.7)
        prompt_box.next_to(s1_label, DOWN, buff=0.5).shift(LEFT * 4)

        # K=4 outputs
        outputs = VGroup(*[_card(f"y{i+1}", "#d2a8ff") for i in range(4)])
        outputs.arrange(DOWN, buff=0.2)
        outputs.next_to(prompt_box, RIGHT, buff=1.5)

        arrows_out = VGroup(*[
            Arrow(prompt_box.get_right(), o.get_left(), buff=0.1,
                  color="#c9d1d9", stroke_width=1.5, max_tip_length_to_length_ratio=0.15)
            for o in outputs
        ])

        k_label = Text("K = 4", font_size=22, color="#d2a8ff", font=_FONT)
        k_label.next_to(outputs, RIGHT, buff=0.4)

        self.play(FadeIn(prompt_box, shift=RIGHT * 0.2), run_time=0.5)
        self.play(
            *[Create(a) for a in arrows_out],
            *[FadeIn(o, shift=RIGHT * 0.2) for o in outputs],
            run_time=0.8
        )
        self.play(FadeIn(k_label), run_time=0.4)

        # ── Labeler ranks them ──
        rank_label = Text("Labeler ranks: y3 > y1 > y4 > y2",
                          font_size=22, color="#f0883e", font=_FONT)
        rank_label.next_to(outputs, DOWN, buff=0.5).shift(LEFT * 0.5)

        # Show ranking numbers
        ranks = ["2", "4", "1", "3"]  # y1=2nd, y2=4th, y3=1st, y4=3rd
        rank_badges = VGroup()
        for i, (o, r) in enumerate(zip(outputs, ranks)):
            badge = Text(f"#{r}", font_size=16, color="#f0883e", weight=BOLD, font=_FONT)
            badge.next_to(o, LEFT, buff=0.15)
            rank_badges.add(badge)

        self.play(FadeIn(rank_label, shift=UP * 0.2), run_time=0.5)
        self.play(*[FadeIn(b) for b in rank_badges], run_time=0.5)
        self.wait(0.8)

        s1_all = VGroup(s1_label, prompt_box, outputs, arrows_out, k_label,
                        rank_label, rank_badges)
        self.play(FadeOut(s1_all), run_time=0.4)

        # ══════════════════════════════════════════════════════════════
        # SCENE 2: Extract C(K,2) pairs
        # ══════════════════════════════════════════════════════════════
        s2_label = Text("Step 2: Extract C(K,2) = 6 comparison pairs",
                        font_size=26, color="#f0883e", weight=BOLD, font=_FONT)
        s2_label.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(s2_label, shift=RIGHT * 0.3))

        # Show all 6 pairs from K=4
        pairs_data = [
            ("y3 > y1", "#3fb950"), ("y3 > y4", "#3fb950"), ("y3 > y2", "#3fb950"),
            ("y1 > y4", "#3fb950"), ("y1 > y2", "#3fb950"), ("y4 > y2", "#3fb950"),
        ]
        pair_cards = VGroup()
        for txt, c in pairs_data:
            card = _card(txt, c, width=1.4, height=0.6)
            pair_cards.add(card)
        pair_cards.arrange_in_grid(rows=2, cols=3, buff=0.3)
        pair_cards.next_to(s2_label, DOWN, buff=0.5)

        formula = MathTex(
            r"\binom{K}{2} = \binom{4}{2} = 6 \text{ pairs}",
            font_size=28, color="#f0883e"
        )
        formula.next_to(pair_cards, DOWN, buff=0.4)

        for card in pair_cards:
            self.play(FadeIn(card, shift=UP * 0.15), run_time=0.25)
        self.play(Write(formula), run_time=0.7)
        self.wait(0.8)

        s2_all = VGroup(s2_label, pair_cards, formula)
        self.play(FadeOut(s2_all), run_time=0.4)

        # ══════════════════════════════════════════════════════════════
        # SCENE 3: Naive vs Batched approach
        # ══════════════════════════════════════════════════════════════
        s3_label = Text("Naive vs Batched Training",
                        font_size=26, color=WHITE, weight=BOLD, font=_FONT)
        s3_label.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(s3_label, shift=RIGHT * 0.3))

        # Left: Naive
        naive_title = Text("Naive", font_size=24, color="#f85149",
                            weight=BOLD, font=_FONT)
        naive_items = VGroup(
            Text("Each pair = separate data point", font_size=18,
                 color=_SUB, font=_FONT),
            Text("6 gradient steps from 1 prompt", font_size=18,
                 color=_SUB, font=_FONT),
            Text("Same y scored 3× → overfitting", font_size=18,
                 color="#f85149", font=_FONT),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        naive_col = VGroup(naive_title, naive_items).arrange(DOWN, buff=0.3)

        # Right: Batched (InstructGPT)
        batch_title = Text("Batched (InstructGPT)", font_size=24, color="#3fb950",
                            weight=BOLD, font=_FONT)
        batch_items = VGroup(
            Text("All C(K,2) pairs = 1 batch element", font_size=18,
                 color=_SUB, font=_FONT),
            Text("1 gradient step from 1 prompt", font_size=18,
                 color=_SUB, font=_FONT),
            Text("1 forward pass per completion", font_size=18,
                 color="#3fb950", font=_FONT),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        batch_col = VGroup(batch_title, batch_items).arrange(DOWN, buff=0.3)

        cols = VGroup(naive_col, batch_col).arrange(RIGHT, buff=1.5)
        cols.next_to(s3_label, DOWN, buff=0.5)

        divider = Line(cols.get_top() + UP * 0.1, cols.get_bottom() + DOWN * 0.1,
                        color=_SUB, stroke_width=1)

        self.play(FadeIn(naive_title), FadeIn(batch_title), Create(divider), run_time=0.6)
        for n_item, b_item in zip(naive_items, batch_items):
            self.play(FadeIn(n_item, shift=RIGHT * 0.2), FadeIn(b_item, shift=LEFT * 0.2),
                      run_time=0.5)
        self.wait(0.5)

        # The loss equation
        loss_eq = MathTex(
            r"\text{loss}(\theta) = -\frac{1}{\binom{K}{2}}"
            r"\sum_{(w,l)\,:\,y_w \succ y_l}"
            r"\log\sigma\!\left(r_\theta(x,y_w) - r_\theta(x,y_l)\right)",
            font_size=26, color=WHITE
        )
        loss_eq.next_to(cols, DOWN, buff=0.5)
        self.play(Write(loss_eq), run_time=1.0)
        self.wait(1.5)

        # ── K scaling note ──
        s3_all = VGroup(s3_label, naive_col, batch_col, divider, loss_eq)
        self.play(FadeOut(s3_all), run_time=0.4)

        # Summary: K scaling
        sum_title = Text("Scaling K Across Dataset", font_size=30,
                         weight=BOLD, color=WHITE, font=_FONT)
        sum_title.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(sum_title))

        k_data = VGroup(
            Text("K = 4  →  C(4,2) =  6 pairs", font_size=22, color="#58a6ff", font=_FONT),
            Text("K = 5  →  C(5,2) = 10 pairs", font_size=22, color="#58a6ff", font=_FONT),
            Text("K = 6  →  C(6,2) = 15 pairs", font_size=22, color="#58a6ff", font=_FONT),
            Text("K = 9  →  C(9,2) = 36 pairs", font_size=22, color="#d2a8ff", font=_FONT),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        k_data.next_to(sum_title, DOWN, buff=0.5)

        note = Text("33K prompts × avg ~20 pairs each = ~660K comparison pairs",
                     font_size=20, color="#3fb950", font=_FONT)
        note.next_to(k_data, DOWN, buff=0.5)

        for item in k_data:
            self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.4)
        self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)
        self.wait(2.0)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
