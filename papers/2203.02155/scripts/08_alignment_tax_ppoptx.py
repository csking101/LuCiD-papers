# 08 — Alignment Tax & PPO-ptx Fix
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Multi-panel showing NLP benchmark regressions for GPT vs SFT vs PPO vs PPO-ptx
# at 175B. Benchmarks: HellaSwag, SQuAD, DROP, WMT FR→EN
# The visual punch: PPO regresses on all, PPO-ptx recovers.
#
# Run:
#   python 08_alignment_tax_ppoptx.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.style import apply_style, COLORS
from shared.plotly_utils import save_plotly_html

apply_style()

OUTPUT_DIR_STATIC = Path(__file__).parent.parent / "output" / "static"
OUTPUT_DIR_INTERACTIVE = Path(__file__).parent.parent / "output" / "interactive"
OUTPUT_DIR_STATIC.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_INTERACTIVE.mkdir(parents=True, exist_ok=True)

# ── Data from Table 14 (175B models) ──
benchmarks = ["HellaSwag\n(acc)", "SQuAD 2\n(F1)", "DROP\n(F1)", "WMT FR→EN\n(BLEU)"]
benchmarks_flat = ["HellaSwag (acc)", "SQuAD 2 (F1)", "DROP (F1)", "WMT FR→EN (BLEU)"]

# Scores at 175B
gpt3_scores = [78.6, 69.0, 36.7, 32.6]
sft_scores = [75.5, 62.5, 31.2, 24.8]
ppo_scores = [71.4, 49.2, 24.0, 18.4]
ppoptx_scores = [78.8, 65.8, 36.5, 33.8]

models = ["GPT-3", "SFT", "PPO", "PPO-ptx"]
model_colors_mpl = [COLORS['grey'], COLORS['blue'], COLORS['red'], COLORS['green']]
model_colors_plotly = ['#6e7681', '#58a6ff', '#f85149', '#3fb950']

# ── Matplotlib static: 2x2 panel ──
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

all_scores = [gpt3_scores, sft_scores, ppo_scores, ppoptx_scores]

for idx, (bench, ax) in enumerate(zip(benchmarks, axes)):
    scores = [s[idx] for s in all_scores]
    x = np.arange(len(models))
    bars = ax.bar(x, scores, color=model_colors_mpl, edgecolor='white',
                  linewidth=0.5, width=0.6)

    # Highlight GPT-3 baseline
    ax.axhline(y=gpt3_scores[idx], color=COLORS['grey'], linestyle='--',
               alpha=0.5, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_title(bench.replace('\n', ' '), fontsize=11, fontweight='bold')

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Mark the drop and recovery
    if scores[2] < scores[0] * 0.85:  # significant drop
        drop = scores[0] - scores[2]
        ax.annotate(f'↓{drop:.1f}', xy=(2, scores[2]),
                    xytext=(2.3, scores[2] - 3),
                    fontsize=9, color=COLORS['red'], fontweight='bold')

fig.suptitle('Alignment Tax: PPO Regresses on NLP Benchmarks, PPO-ptx Recovers',
             fontsize=13, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTPUT_DIR_STATIC / "08_alignment_tax_ppoptx.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '08_alignment_tax_ppoptx.png'}")

# ── Plotly interactive: grouped bar ──
fig_plotly = go.Figure()

for model, scores, color in zip(models, all_scores, model_colors_plotly):
    fig_plotly.add_trace(go.Bar(
        x=benchmarks_flat, y=scores, name=model,
        marker_color=color,
        text=[f'{v:.1f}' for v in scores],
        textposition='outside', textfont=dict(size=11),
        hovertemplate=f'{model}<br>%{{x}}<br>Score: %{{y:.1f}}<extra></extra>'
    ))

# Add GPT-3 baseline reference annotations
for i, (bench, gpt_score) in enumerate(zip(benchmarks_flat, gpt3_scores)):
    fig_plotly.add_annotation(
        x=bench, y=gpt_score + 2,
        text=f"GPT-3: {gpt_score:.1f}",
        showarrow=False,
        font=dict(color="#6e7681", size=10),
        yshift=15
    )

fig_plotly.update_layout(
    template="plotly_dark",
    title=dict(
        text="Alignment Tax: PPO Regresses, PPO-ptx Recovers (175B)",
        font=dict(size=16)
    ),
    yaxis=dict(title="Score"),
    margin=dict(t=60, b=80, l=50, r=30),
    barmode='group',
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(22,27,34,0.8)"),
)

save_plotly_html(fig_plotly, OUTPUT_DIR_INTERACTIVE / "08_alignment_tax_ppoptx.html")
