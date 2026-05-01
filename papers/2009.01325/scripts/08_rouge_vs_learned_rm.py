# 08 — ROUGE vs Learned Reward Model
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Comparison of ROUGE, BLEU, and learned reward model agreement with human evaluators.
# Shows learned RM wins even on out-of-distribution CNN/DM dataset.
#
# Run:
#   python 08_rouge_vs_learned_rm.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from shared.style import apply_style, COLORS
from shared.plotly_utils import save_plotly_html

apply_style()

OUTPUT_DIR_STATIC = Path(__file__).parent.parent / "output" / "static"
OUTPUT_DIR_INTERACTIVE = Path(__file__).parent.parent / "output" / "interactive"
OUTPUT_DIR_STATIC.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_INTERACTIVE.mkdir(parents=True, exist_ok=True)

# ── Data: metric agreement with human preferences (approximate from paper) ──
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'Learned RM\n(1.3B)', 'Learned RM\n(6.7B)']
metrics_clean = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'Learned RM (1.3B)', 'Learned RM (6.7B)']

# Agreement rates on TL;DR (%)
agreement_tldr = [57.0, 56.5, 57.2, 53.0, 62.4, 65.8]
# Agreement rates on CNN/DM (transfer, no training on news)
agreement_cnn = [55.5, 55.0, 55.8, 52.0, 60.2, 63.5]
# Human inter-annotator agreement
human_agree = 66.9

bar_colors = [COLORS['grey'], COLORS['grey'], COLORS['grey'], COLORS['grey'],
              COLORS['green'], COLORS['green']]

# ── Matplotlib static ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

x = np.arange(len(metrics))
width = 0.6

# TL;DR
bars1 = ax1.bar(x, agreement_tldr, width, color=bar_colors, edgecolor='white', linewidth=0.5)
ax1.axhline(y=human_agree, color=COLORS['orange'], linestyle='--', alpha=0.7,
            label=f'Human agreement ({human_agree}%)')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=8)
ax1.set_ylabel('Agreement with Humans (%)')
ax1.set_title('TL;DR (in-distribution)')
ax1.set_ylim(48, 72)
ax1.legend(fontsize=8)
for bar, val in zip(bars1, agreement_tldr):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# CNN/DM
bars2 = ax2.bar(x, agreement_cnn, width, color=bar_colors, edgecolor='white', linewidth=0.5)
ax2.axhline(y=human_agree, color=COLORS['orange'], linestyle='--', alpha=0.7,
            label=f'Human agreement ({human_agree}%)')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=8)
ax2.set_title('CNN/DM (out-of-distribution)')
ax2.legend(fontsize=8)
for bar, val in zip(bars2, agreement_cnn):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

fig.suptitle('Automatic Metrics vs Learned Reward Model: Agreement with Humans',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR_STATIC / "08_rouge_vs_learned_rm.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '08_rouge_vs_learned_rm.png'}")

# ── Plotly interactive ──
fig_plotly = go.Figure()

fig_plotly.add_trace(go.Bar(
    x=metrics_clean, y=agreement_tldr, name='TL;DR (in-dist.)',
    marker_color=['#95A5A6'] * 4 + ['#27AE60'] * 2,
    text=[f'{v:.1f}%' for v in agreement_tldr], textposition='outside',
    textfont=dict(size=11),
))
fig_plotly.add_trace(go.Bar(
    x=metrics_clean, y=agreement_cnn, name='CNN/DM (out-of-dist.)',
    marker_color=['#6c757d'] * 4 + ['#1a8a42'] * 2,
    text=[f'{v:.1f}%' for v in agreement_cnn], textposition='outside',
    textfont=dict(size=11),
))

fig_plotly.add_hline(
    y=human_agree, line_dash="dash", line_color="#E8790C", line_width=2,
    annotation_text=f"Human agreement ({human_agree}%)",
    annotation_position="top right",
    annotation_font=dict(color="#E8790C", size=12)
)

fig_plotly.update_layout(
    template="plotly_dark",
    title=dict(text="Automatic Metrics vs Learned RM: Agreement with Humans", font=dict(size=17)),
    yaxis=dict(title="Agreement with Humans (%)", range=[48, 72]),
    barmode='group',
    margin=dict(t=60, b=80, l=50, r=30),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(22,27,34,0.8)"),
)

save_plotly_html(fig_plotly, OUTPUT_DIR_INTERACTIVE / "08_rouge_vs_learned_rm.html")
