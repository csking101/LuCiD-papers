# 06 — TruthfulQA Results
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Grouped bar chart from Table 14 data:
# Truthful & Truthful+Informative fractions across GPT/SFT/PPO/PPO-ptx
# at multiple sizes. PPO shows ~2x improvement over GPT-3.
#
# Run:
#   python 06_truthfulqa_results.py

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

# ── Data from Table 14 / Figure 5 (approximate from paper) ──
# Models at 175B size
models_175b = ["GPT-3\n175B", "GPT-3\n175B\n(prompted)", "SFT\n175B", "PPO\n175B", "PPO-ptx\n175B"]

# Truthful (% of responses that are truthful)
truthful = [0.22, 0.28, 0.29, 0.47, 0.50]
# Truthful + Informative (% that are both truthful AND informative)
truthful_informative = [0.19, 0.25, 0.24, 0.36, 0.41]

# Also show scaling: 1.3B and 6B
models_all = [
    "GPT 1.3B", "GPT 6B", "GPT 175B",
    "SFT 1.3B", "SFT 6B", "SFT 175B",
    "PPO 1.3B", "PPO 6B", "PPO 175B",
    "PPO-ptx 1.3B", "PPO-ptx 6B", "PPO-ptx 175B",
]
truthful_all = [0.37, 0.30, 0.22, 0.43, 0.36, 0.29, 0.52, 0.51, 0.47, 0.53, 0.51, 0.50]
truthful_inf_all = [0.27, 0.22, 0.19, 0.35, 0.28, 0.24, 0.42, 0.40, 0.36, 0.44, 0.42, 0.41]

# ── Matplotlib static: 175B comparison ──
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(models_175b))
width = 0.35

b1 = ax.bar(x - width/2, truthful, width, label='Truthful',
            color=COLORS['blue'], edgecolor='white', linewidth=0.5)
b2 = ax.bar(x + width/2, truthful_informative, width, label='Truthful + Informative',
            color=COLORS['green'], edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(models_175b, fontsize=9, ha='center')
ax.set_ylabel('Fraction of Responses')
ax.set_title('TruthfulQA: InstructGPT vs GPT-3 at 175B')
ax.set_ylim(0, 0.65)
ax.legend(fontsize=9)

for bar_group in [b1, b2]:
    for bar in bar_group:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=9)

fig.tight_layout()
fig.savefig(OUTPUT_DIR_STATIC / "06_truthfulqa_results.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '06_truthfulqa_results.png'}")

# ── Plotly interactive: all sizes with toggle ──
fig_plotly = go.Figure()

# 175B view (default)
fig_plotly.add_trace(go.Bar(
    x=[m.replace('\n', ' ') for m in models_175b],
    y=truthful, name="Truthful",
    marker_color='#58a6ff',
    text=[f'{v:.0%}' for v in truthful],
    textposition='outside', textfont=dict(size=12),
    hovertemplate='%{x}<br>Truthful: %{y:.1%}<extra></extra>',
    visible=True
))
fig_plotly.add_trace(go.Bar(
    x=[m.replace('\n', ' ') for m in models_175b],
    y=truthful_informative, name="Truthful + Informative",
    marker_color='#3fb950',
    text=[f'{v:.0%}' for v in truthful_informative],
    textposition='outside', textfont=dict(size=12),
    hovertemplate='%{x}<br>Truthful+Informative: %{y:.1%}<extra></extra>',
    visible=True
))

# All sizes view
fig_plotly.add_trace(go.Bar(
    x=models_all, y=truthful_all, name="Truthful (all)",
    marker_color='#58a6ff',
    text=[f'{v:.0%}' for v in truthful_all],
    textposition='outside', textfont=dict(size=10),
    hovertemplate='%{x}<br>Truthful: %{y:.1%}<extra></extra>',
    visible=False
))
fig_plotly.add_trace(go.Bar(
    x=models_all, y=truthful_inf_all, name="T+I (all)",
    marker_color='#3fb950',
    text=[f'{v:.0%}' for v in truthful_inf_all],
    textposition='outside', textfont=dict(size=10),
    hovertemplate='%{x}<br>Truthful+Informative: %{y:.1%}<extra></extra>',
    visible=False
))

fig_plotly.update_layout(
    template="plotly_dark",
    title=dict(text="TruthfulQA: InstructGPT vs GPT-3", font=dict(size=18)),
    yaxis=dict(title="Fraction of Responses", range=[0, 0.65]),
    margin=dict(t=60, b=100, l=50, r=30),
    barmode='group',
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(22,27,34,0.8)"),
    updatemenus=[dict(
        type="buttons",
        direction="right",
        x=0.5, xanchor="center", y=-0.18,
        buttons=[
            dict(label="175B Only",
                 method="update",
                 args=[{"visible": [True, True, False, False]},
                       {"xaxis.tickangle": 0}]),
            dict(label="All Sizes",
                 method="update",
                 args=[{"visible": [False, False, True, True]},
                       {"xaxis.tickangle": -45}]),
        ],
        font=dict(size=12),
        bgcolor="#21262d",
    )]
)

save_plotly_html(fig_plotly, OUTPUT_DIR_INTERACTIVE / "06_truthfulqa_results.html")
