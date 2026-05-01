# 04 — SFT vs RL Quality Comparison
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Bar chart showing human preference rates for SFT baselines vs RL policies.
# Interactive version lets you toggle length-controlled results.
#
# Run:
#   python 04_sft_vs_rl_quality.py

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

# ── Data from paper (approximate from Figure 1 / Table 1) ──
models = [
    "Pretrained\n(zero-shot)",
    "SFT\n(1.3B)",
    "SFT\n(6.7B)",
    "RL\n(1.3B)",
    "RL\n(6.7B)",
    "Human\nReference",
]
# Win rate vs human reference (approximate percentages)
win_rates = [15, 35, 42, 57, 70, 50]
win_rates_length_ctrl = [12, 32, 38, 52, 65, 50]

colors_bar = [COLORS['grey'], COLORS['blue'], COLORS['blue'],
              COLORS['green'], COLORS['green'], COLORS['orange']]

# ── Matplotlib static ──
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models))
bars = ax.bar(x, win_rates, color=colors_bar, edgecolor='white', linewidth=0.5, width=0.6)
ax.axhline(y=50, color=COLORS['orange'], linestyle='--', alpha=0.5, label='Human reference baseline')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('Human Preference Rate (%)')
ax.set_title('SFT vs RL Policies: Human Preference on TL;DR')
ax.set_ylim(0, 85)
ax.legend(fontsize=9)

for bar, val in zip(bars, win_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

fig.tight_layout()
fig.savefig(OUTPUT_DIR_STATIC / "04_sft_vs_rl_quality.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '04_sft_vs_rl_quality.png'}")

# ── Plotly interactive ──
model_labels = ["Pretrained (zero-shot)", "SFT (1.3B)", "SFT (6.7B)",
                "RL (1.3B)", "RL (6.7B)", "Human Reference"]
plotly_colors = ['#95A5A6', '#2E86C1', '#2E86C1', '#27AE60', '#27AE60', '#E8790C']

fig_plotly = go.Figure()

# Unconstrained bars
fig_plotly.add_trace(go.Bar(
    x=model_labels, y=win_rates, name="Unconstrained",
    marker_color=plotly_colors, text=[f"{v}%" for v in win_rates],
    textposition='outside', textfont=dict(size=13),
    visible=True
))

# Length-controlled bars
fig_plotly.add_trace(go.Bar(
    x=model_labels, y=win_rates_length_ctrl, name="Length-Controlled",
    marker_color=[c.replace(')', ', 0.7)').replace('rgb', 'rgba') if 'rgb' in c else c
                     for c in plotly_colors],
    text=[f"{v}%" for v in win_rates_length_ctrl],
    textposition='outside', textfont=dict(size=13),
    visible=False
))

# Reference line (always visible, two copies)
for vis in [True, False]:
    fig_plotly.add_hline(y=50, line_dash="dash", line_color="#E8790C",
                          opacity=0.5, annotation_text="Human reference" if vis else None)

fig_plotly.update_layout(
    template="plotly_dark",
    title=dict(text="SFT vs RL: Human Preference on TL;DR", font=dict(size=18)),
    yaxis=dict(title="Human Preference Rate (%)", range=[0, 85]),
    margin=dict(t=60, b=80, l=50, r=30),
    showlegend=False,
    updatemenus=[dict(
        type="buttons",
        direction="right",
        x=0.5, xanchor="center", y=-0.15,
        buttons=[
            dict(label="Unconstrained",
                 method="update",
                 args=[{"visible": [True, False]}]),
            dict(label="Length-Controlled",
                 method="update",
                 args=[{"visible": [False, True]}]),
        ],
        font=dict(size=12),
        bgcolor="#21262d",
    )]
)

save_plotly_html(fig_plotly, OUTPUT_DIR_INTERACTIVE / "04_sft_vs_rl_quality.html")
