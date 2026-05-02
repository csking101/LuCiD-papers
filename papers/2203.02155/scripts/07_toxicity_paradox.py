# 07 — Toxicity Paradox
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Three prompt conditions (respectful / basic / biased) showing toxicity scores.
# The paradox: InstructGPT is LESS toxic with respectful prompts,
# but MORE toxic than GPT-3 when explicitly prompted to be biased/toxic.
#
# Run:
#   python 07_toxicity_paradox.py

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

# ── Data from Table 14 / Figure 6 (approximate from paper) ──
# Toxicity score (Perspective API) for 175B models under 3 prompt conditions
prompt_conditions = ["Respectful\nPrompt", "Standard\nPrompt", "Biased\nPrompt"]
prompt_conditions_flat = ["Respectful Prompt", "Standard Prompt", "Biased Prompt"]

# Average maximum toxicity scores (0-1 scale, higher = more toxic)
gpt3_toxicity = [0.42, 0.49, 0.53]
sft_toxicity = [0.35, 0.40, 0.55]
ppo_toxicity = [0.22, 0.33, 0.60]
ppoptx_toxicity = [0.20, 0.32, 0.62]

models = ["GPT-3 175B", "SFT 175B", "PPO 175B", "PPO-ptx 175B"]
model_colors_mpl = [COLORS['grey'], COLORS['blue'], COLORS['green'], COLORS['teal']]
model_colors_plotly = ['#6e7681', '#58a6ff', '#3fb950', '#1abc9c']

# ── Matplotlib static ──
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(prompt_conditions))
width = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

all_toxicities = [gpt3_toxicity, sft_toxicity, ppo_toxicity, ppoptx_toxicity]

for i, (model, tox, color) in enumerate(zip(models, all_toxicities, model_colors_mpl)):
    bars = ax.bar(x + offsets[i] * width, tox, width, label=model,
                  color=color, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, tox):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(prompt_conditions, fontsize=10)
ax.set_ylabel('Average Maximum Toxicity')
ax.set_title('Toxicity Paradox: InstructGPT Follows Instructions — Even Harmful Ones')
ax.set_ylim(0, 0.8)
ax.legend(fontsize=9, loc='upper left')

# Add annotation for the paradox
ax.annotate('Paradox: RLHF models\nmore toxic when prompted\nto be biased',
            xy=(2.3, 0.62), xytext=(2.0, 0.75),
            arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
            fontsize=9, color=COLORS['red'], fontweight='bold',
            ha='center')

# Add shading zones
ax.axvspan(-0.5, 0.5, alpha=0.05, color='green', label='_nolegend_')
ax.axvspan(1.5, 2.5, alpha=0.05, color='red', label='_nolegend_')

fig.tight_layout()
fig.savefig(OUTPUT_DIR_STATIC / "07_toxicity_paradox.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '07_toxicity_paradox.png'}")

# ── Plotly interactive ──
fig_plotly = go.Figure()

for model, tox, color in zip(models, all_toxicities, model_colors_plotly):
    fig_plotly.add_trace(go.Bar(
        x=prompt_conditions_flat, y=tox, name=model,
        marker_color=color,
        text=[f'{v:.2f}' for v in tox],
        textposition='outside', textfont=dict(size=11),
        hovertemplate=f'{model}<br>%{{x}}<br>Toxicity: %{{y:.3f}}<extra></extra>'
    ))

# Paradox annotation
fig_plotly.add_annotation(
    x="Biased Prompt", y=0.67,
    text="<b>Paradox</b>: RLHF models more toxic<br>when prompted to be biased",
    showarrow=True, arrowhead=2, arrowcolor="#f85149",
    font=dict(color="#f85149", size=12),
    yshift=20
)

# Shaded regions
fig_plotly.add_vrect(x0=-0.5, x1=0.5, fillcolor="#3fb950", opacity=0.05,
                     line_width=0, annotation_text="Aligned ✓",
                     annotation_position="top left",
                     annotation_font=dict(color="#3fb950", size=11))
fig_plotly.add_vrect(x0=1.5, x1=2.5, fillcolor="#f85149", opacity=0.05,
                     line_width=0, annotation_text="Misaligned ✗",
                     annotation_position="top right",
                     annotation_font=dict(color="#f85149", size=11))

fig_plotly.update_layout(
    template="plotly_dark",
    title=dict(text="Toxicity Paradox: InstructGPT Follows Instructions — Even Harmful Ones",
               font=dict(size=16)),
    yaxis=dict(title="Average Maximum Toxicity (Perspective API)", range=[0, 0.82]),
    margin=dict(t=60, b=80, l=50, r=30),
    barmode='group',
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(22,27,34,0.8)"),
)

save_plotly_html(fig_plotly, OUTPUT_DIR_INTERACTIVE / "07_toxicity_paradox.html")
