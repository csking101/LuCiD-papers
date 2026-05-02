# 09 — Who Are We Aligning To?
#
# Tool: Matplotlib
# Output: PNG (static infographic)
#
# Four concentric layers showing the alignment hierarchy:
# Innermost: Labelers (who provide the data)
# Then: Researchers (who design the process)
# Then: Customers (who use the API)
# Outermost: End Users (affected by outputs)
#
# With callout annotations about biases/limitations at each layer.
#
# Run:
#   python 09_who_are_we_aligning_to.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from shared.style import apply_style, COLORS

apply_style()

OUTPUT_DIR_STATIC = Path(__file__).parent.parent / "output" / "static"
OUTPUT_DIR_STATIC.mkdir(parents=True, exist_ok=True)

# ── Layer definitions ──
layers = [
    {
        "label": "Labelers",
        "radius": 1.0,
        "color": "#58a6ff",
        "bias": "~40 contractors\nEnglish-speaking\nSkewed demographics\nScreened for agreement",
    },
    {
        "label": "Researchers",
        "radius": 2.0,
        "color": "#d2a8ff",
        "bias": "Define the task\nWrite instructions\nChoose labeler pool\nDesign reward criteria",
    },
    {
        "label": "Customers",
        "radius": 3.0,
        "color": "#f0883e",
        "bias": "API users (developers)\nBusiness objectives\nMay not represent\nend user interests",
    },
    {
        "label": "End Users",
        "radius": 4.0,
        "color": "#3fb950",
        "bias": "Affected by outputs\nNo direct voice\nin training process\nMost diverse group",
    },
]

# ── Build the figure ──
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-5.5, 5.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0d1117')

# Title
ax.text(0, 5.0, "Who Are We Aligning To?",
        fontsize=22, fontweight='bold', color='white',
        ha='center', va='center',
        fontfamily='serif')
ax.text(0, 4.4, "The four-layer alignment hierarchy in InstructGPT",
        fontsize=13, color='#c9d1d9', ha='center', va='center',
        fontfamily='serif')

# Draw concentric rings (outermost first for proper layering)
for layer in reversed(layers):
    circle = plt.Circle(
        (0, 0), layer["radius"],
        fill=True, facecolor=layer["color"],
        alpha=0.08, edgecolor=layer["color"],
        linewidth=2.0, linestyle='-'
    )
    ax.add_patch(circle)

# Add labels inside each ring
for i, layer in enumerate(layers):
    r = layer["radius"]
    # Place label at the ring boundary, slightly inward
    if i == 0:
        label_y = 0
    else:
        label_y = (layers[i-1]["radius"] + r) / 2

    ax.text(0, label_y, layer["label"],
            fontsize=14 - i, fontweight='bold',
            color=layer["color"], ha='center', va='center',
            fontfamily='serif',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='#0d1117', edgecolor=layer["color"],
                      alpha=0.9, linewidth=1.5))

# Add callout annotations
annotation_positions = [
    # (text_x, text_y, arrow_target_x, arrow_target_y)
    (-5.0, 1.5, -0.8, 0.3),     # Labelers - left
    (5.0, 2.5, 1.3, 1.2),       # Researchers - right
    (-5.0, -1.5, -2.2, -1.5),   # Customers - left
    (5.0, -2.0, 3.0, -2.0),     # End Users - right
]

for layer, (tx, ty, ax_x, ax_y) in zip(layers, annotation_positions):
    ax.annotate(
        layer["bias"],
        xy=(ax_x, ax_y),
        xytext=(tx, ty),
        fontsize=10, color=layer["color"],
        fontfamily='serif',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='#161b22', edgecolor=layer["color"],
                  alpha=0.95, linewidth=1.5),
        arrowprops=dict(
            arrowstyle='->', color=layer["color"],
            lw=1.5, connectionstyle='arc3,rad=0.2'
        )
    )

# Direction arrow showing "training signal flows inward"
ax.annotate(
    '', xy=(0, -1.0), xytext=(0, -3.8),
    arrowprops=dict(arrowstyle='->', color='#c9d1d9',
                    lw=2, connectionstyle='arc3,rad=0')
)
ax.text(0.3, -2.4, "Training signal\nflows inward",
        fontsize=10, color='#c9d1d9', fontfamily='serif',
        ha='left', va='center', style='italic')

# Key insight at bottom
ax.text(0, -5.0,
        "Key limitation: We align to labeler preferences, not end user preferences",
        fontsize=12, fontweight='bold', color='#f85149',
        ha='center', va='center', fontfamily='serif',
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='#161b22', edgecolor='#f85149',
                  alpha=0.95, linewidth=1.5))

fig.savefig(OUTPUT_DIR_STATIC / "09_who_are_we_aligning_to.png", dpi=200,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '09_who_are_we_aligning_to.png'}")
