# 05 — Winrate vs Baselines
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Grouped bar chart reproducing Figure 1 from the paper:
# Winrate against 175B SFT baseline for all model variants.
# Headline result: 1.3B PPO-ptx preferred over 175B GPT-3
#
# Run:
#   python 05_winrate_vs_baselines.py

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

# ── Data from Figure 1 (approximate from paper) ──
# Winrate against 175B SFT baseline on API distribution
# Format: (model_label, winrate, color_key)
models_data = [
    # GPT baselines
    ("GPT\n1.3B", 18, "grey"),
    ("GPT\n6B", 22, "grey"),
    ("GPT\n175B", 29, "grey"),
    # GPT + prompted
    ("GPT\n175B\n(prompted)", 36, "light_blue"),
    # SFT
    ("SFT\n1.3B", 37, "blue"),
    ("SFT\n6B", 43, "blue"),
    ("SFT\n175B", 50, "blue"),  # reference
    # PPO
    ("PPO\n1.3B", 56, "green"),
    ("PPO\n6B", 62, "green"),
    ("PPO\n175B", 68, "green"),
    # PPO-ptx
    ("PPO-ptx\n1.3B", 55, "teal"),
    ("PPO-ptx\n6B", 61, "teal"),
    ("PPO-ptx\n175B", 71, "teal"),
]

model_labels = [m[0] for m in models_data]
winrates = [m[1] for m in models_data]
bar_colors_mpl = [COLORS[m[2]] for m in models_data]

# Plotly-specific flat labels
model_labels_flat = [m[0].replace('\n', ' ') for m in models_data]

# ── Matplotlib static ──
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(models_data))
bars = ax.bar(x, winrates, color=bar_colors_mpl, edgecolor='white',
              linewidth=0.5, width=0.7)

ax.axhline(y=50, color=COLORS['blue'], linestyle='--', alpha=0.5,
           label='175B SFT baseline')

# Highlight the 1.3B PPO-ptx > 175B GPT-3 result
ax.annotate('1.3B InstructGPT > 175B GPT-3',
            xy=(10, 55), xytext=(7, 75),
            arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=2),
            fontsize=11, color=COLORS['orange'], fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=8, ha='center')
ax.set_ylabel('Winrate vs 175B SFT (%)')
ax.set_title('InstructGPT: Human Preference Winrates (Figure 1)')
ax.set_ylim(0, 85)
ax.legend(fontsize=9)

for bar, val in zip(bars, winrates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.tight_layout()
fig.savefig(OUTPUT_DIR_STATIC / "05_winrate_vs_baselines.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '05_winrate_vs_baselines.png'}")

# ── Plotly interactive ──
plotly_colors = {
    'grey': '#6e7681', 'light_blue': '#85c1e9', 'blue': '#58a6ff',
    'green': '#3fb950', 'teal': '#1abc9c'
}
bar_colors_plotly = [plotly_colors[m[2]] for m in models_data]

fig_plotly = go.Figure()

fig_plotly.add_trace(go.Bar(
    x=model_labels_flat, y=winrates,
    marker_color=bar_colors_plotly,
    text=[f'{v}%' for v in winrates],
    textposition='outside', textfont=dict(size=11),
    hovertemplate='%{x}<br>Winrate: %{y}%<extra></extra>'
))

# Reference line at 50%
fig_plotly.add_hline(
    y=50, line_dash="dash", line_color="#58a6ff", line_width=2,
    annotation_text="175B SFT baseline",
    annotation_position="top right",
    annotation_font=dict(color="#58a6ff", size=12)
)

# Highlight annotation
fig_plotly.add_annotation(
    x="PPO-ptx 1.3B", y=55, ax="GPT 175B", ay=29,
    xref="x", yref="y", axref="x", ayref="y",
    showarrow=True, arrowhead=2, arrowsize=1.5,
    arrowcolor="#f0883e", arrowwidth=2,
    text="<b>1.3B InstructGPT<br>> 175B GPT-3</b>",
    font=dict(color="#f0883e", size=12),
    xshift=0, yshift=25
)

fig_plotly.update_layout(
    template="plotly_dark",
    title=dict(text="InstructGPT: Winrate vs 175B SFT Baseline (Figure 1)", font=dict(size=18)),
    yaxis=dict(title="Winrate (%)", range=[0, 85]),
    xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
    margin=dict(t=60, b=120, l=50, r=30),
    showlegend=False,
)

save_plotly_html(fig_plotly, OUTPUT_DIR_INTERACTIVE / "05_winrate_vs_baselines.html")
