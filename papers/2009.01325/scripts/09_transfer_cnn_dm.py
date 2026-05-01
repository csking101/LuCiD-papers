# 09 — Transfer to CNN/DM
#
# Tool: Matplotlib
# Output: PNG
#
# Shows that models trained only on Reddit TL;DR transfer to CNN/DM news
# summarization without domain-specific training.
#
# Run:
#   python 09_transfer_cnn_dm.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt

from shared.style import apply_style, COLORS

apply_style()

OUTPUT_DIR_STATIC = Path(__file__).parent.parent / "output" / "static"
OUTPUT_DIR_STATIC.mkdir(parents=True, exist_ok=True)

# ── Data: preference rates on CNN/DM (approximate from paper) ──
models = [
    'Lead-3\n(baseline)',
    'SFT\n(6.7B)',
    'RL (TL;DR\ntrained, 6.7B)',
    'T5\n(fine-tuned\non CNN/DM)',
]
# Preference rate vs Lead-3 baseline (approximate)
preference_vs_lead3 = [50, 55, 70, 62]
bar_colors = [COLORS['grey'], COLORS['blue'], COLORS['green'], COLORS['purple']]

fig, ax = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(models))
bars = ax.bar(x, preference_vs_lead3, color=bar_colors, edgecolor='white',
              linewidth=0.5, width=0.55)

ax.axhline(y=50, color=COLORS['grey'], linestyle='--', alpha=0.5, label='Baseline (Lead-3)')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('Human Preference Rate (%)')
ax.set_title('Transfer to CNN/DM: Reddit-Trained RL Model Outperforms\nDomain-Specific Baselines')
ax.set_ylim(0, 85)
ax.legend(fontsize=9, loc='upper left')

for bar, val in zip(bars, preference_vs_lead3):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight the RL bar
bars[2].set_edgecolor(COLORS['green'])
bars[2].set_linewidth(2)

# Add annotation
ax.annotate('Trained only on Reddit!\nNo news-specific data.',
            xy=(2, 70), xytext=(2.8, 78),
            arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1.5),
            fontsize=10, color=COLORS['green'], fontweight='bold',
            ha='center')

fig.tight_layout()
fig.savefig(OUTPUT_DIR_STATIC / "09_transfer_cnn_dm.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '09_transfer_cnn_dm.png'}")
