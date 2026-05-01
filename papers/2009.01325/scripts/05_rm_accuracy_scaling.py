# 05 — Reward Model Accuracy vs Data
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Line plot showing RM accuracy scaling with comparison data size (log-linear).
# Shows ceiling = inter-annotator agreement (66.9%).
#
# Run:
#   python 05_rm_accuracy_scaling.py

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

# ── Synthetic data approximating paper's scaling curves ──
np.random.seed(42)
data_sizes = np.array([1000, 2000, 5000, 10000, 20000, 40000, 65000])
human_agreement = 66.9

# Accuracy curves for different model sizes (approximate log-linear scaling)
def rm_accuracy(n, scale, offset):
    return offset + scale * np.log(n / 1000)

acc_1_3b = rm_accuracy(data_sizes, 3.2, 54.0) + np.random.normal(0, 0.3, len(data_sizes))
acc_6_7b = rm_accuracy(data_sizes, 3.5, 55.5) + np.random.normal(0, 0.3, len(data_sizes))

# Clip to reasonable range
acc_1_3b = np.clip(acc_1_3b, 52, 66)
acc_6_7b = np.clip(acc_6_7b, 53, 67)

# ── Matplotlib static ──
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(data_sizes, acc_1_3b, 'o-', color=COLORS['blue'], linewidth=2,
            markersize=6, label='RM (1.3B)')
ax.semilogx(data_sizes, acc_6_7b, 's-', color=COLORS['green'], linewidth=2,
            markersize=6, label='RM (6.7B)')
ax.axhline(y=human_agreement, color=COLORS['orange'], linestyle='--',
           linewidth=1.5, label=f'Human agreement ({human_agreement}%)')
ax.fill_between([800, 80000], human_agreement - 1, human_agreement + 1,
                color=COLORS['orange'], alpha=0.1)
ax.set_xlabel('Number of Human Comparisons')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Reward Model Accuracy Scales Log-Linearly with Data')
ax.set_xlim(800, 80000)
ax.set_ylim(50, 72)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR_STATIC / "05_rm_accuracy_scaling.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '05_rm_accuracy_scaling.png'}")

# ── Plotly interactive ──
fig_plotly = go.Figure()

fig_plotly.add_trace(go.Scatter(
    x=data_sizes, y=acc_1_3b, mode='lines+markers', name='RM (1.3B)',
    line=dict(color='#2E86C1', width=2.5), marker=dict(size=8)
))
fig_plotly.add_trace(go.Scatter(
    x=data_sizes, y=acc_6_7b, mode='lines+markers', name='RM (6.7B)',
    line=dict(color='#27AE60', width=2.5), marker=dict(size=8, symbol='square')
))
fig_plotly.add_hline(
    y=human_agreement, line_dash="dash", line_color="#E8790C", line_width=2,
    annotation_text=f"Human inter-annotator agreement ({human_agreement}%)",
    annotation_position="top right",
    annotation_font=dict(color="#E8790C", size=12)
)

fig_plotly.update_layout(
    template="plotly_dark",
    title=dict(text="Reward Model Accuracy vs Training Data", font=dict(size=18)),
    xaxis=dict(title="Number of Human Comparisons", type="log",
               tickvals=data_sizes, ticktext=[f"{d//1000}K" for d in data_sizes]),
    yaxis=dict(title="Accuracy (%)", range=[50, 72]),
    margin=dict(t=60, b=80, l=50, r=30),
    legend=dict(x=0.02, y=0.15, bgcolor="rgba(22,27,34,0.8)"),
    hovermode="x unified",
)

save_plotly_html(fig_plotly, OUTPUT_DIR_INTERACTIVE / "05_rm_accuracy_scaling.html")
