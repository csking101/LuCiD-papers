# 04 — Task Distribution & Dataset Sizes
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Two-panel visualization:
# Panel 1: Donut chart of API prompt categories (Table 1)
# Panel 2: Stacked bar of dataset sizes (Table 6) with labeler/customer split
#
# Run:
#   python 04_task_distribution_dataset.py

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

# ── Data from Table 1: API prompt categories ──
categories = [
    "Generation", "Open QA", "Brainstorming", "Chat",
    "Rewrite", "Summarization", "Closed QA", "Classification",
    "Extract", "Other"
]
percentages = [45.6, 12.4, 11.2, 8.4, 6.6, 4.2, 2.6, 3.5, 1.9, 3.6]

cat_colors = [
    '#58a6ff', '#3fb950', '#d2a8ff', '#f0883e',
    '#f85149', '#1abc9c', '#e8790c', '#85c1e9',
    '#f39c12', '#95a5a6'
]

# ── Data from Table 6: Dataset sizes ──
datasets = ["SFT", "RM", "PPO"]
labeler_demos = [12725, 33207, 31144]  # approximate from paper
# Split: labeler-written prompts vs customer API prompts
labeler_prompts = [np.nan, np.nan, np.nan]  # The paper doesn't split SFT this way
# Instead: show total sizes per dataset
# More useful: show prompt sources for the SFT dataset
sft_sources = {"Labeler-written": 11295, "Customer API": 1430}  # approximate
rm_sources = {"Labeler-written": 6623, "Customer API": 26584}   # approximate
ppo_sources = {"Labeler-written": 0, "Customer API": 31144}     # all customer

labeler_counts = [11295, 6623, 0]
customer_counts = [1430, 26584, 31144]

# ── Matplotlib static: two-panel ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Donut chart
wedges, texts, autotexts = ax1.pie(
    percentages, labels=categories, colors=cat_colors,
    autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
    pctdistance=0.8, startangle=90, textprops={'fontsize': 9}
)
centre_circle = plt.Circle((0, 0), 0.5, fc='#fafafa')
ax1.add_artist(centre_circle)
ax1.set_title('API Prompt Categories (Table 1)', fontsize=12, fontweight='bold')

# Panel 2: Stacked bar chart
x = np.arange(len(datasets))
width = 0.5
b1 = ax2.bar(x, labeler_counts, width, label='Labeler-written', color=COLORS['blue'])
b2 = ax2.bar(x, customer_counts, width, bottom=labeler_counts,
             label='Customer API', color=COLORS['green'])

ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=10)
ax2.set_ylabel('Number of Prompts')
ax2.set_title('Dataset Sizes by Source (Table 6)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

# Add total labels
for i, (l, c) in enumerate(zip(labeler_counts, customer_counts)):
    total = l + c
    ax2.text(i, total + 500, f'{total:,}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

fig.tight_layout(w_pad=3)
fig.savefig(OUTPUT_DIR_STATIC / "04_task_distribution_dataset.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '04_task_distribution_dataset.png'}")

# ── Plotly interactive ──
fig_plotly = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "pie"}, {"type": "bar"}]],
    subplot_titles=["API Prompt Categories (Table 1)", "Dataset Sizes by Source (Table 6)"],
    horizontal_spacing=0.12
)

# Donut chart
fig_plotly.add_trace(
    go.Pie(
        labels=categories, values=percentages,
        hole=0.45, marker=dict(colors=cat_colors),
        textinfo='label+percent', textfont=dict(size=11),
        hovertemplate='%{label}: %{percent}<extra></extra>'
    ),
    row=1, col=1
)

# Stacked bar
fig_plotly.add_trace(
    go.Bar(
        x=datasets, y=labeler_counts, name="Labeler-written",
        marker_color='#58a6ff',
        text=[f'{v:,}' for v in labeler_counts], textposition='inside',
        hovertemplate='%{x}<br>Labeler: %{y:,}<extra></extra>'
    ),
    row=1, col=2
)
fig_plotly.add_trace(
    go.Bar(
        x=datasets, y=customer_counts, name="Customer API",
        marker_color='#3fb950',
        text=[f'{v:,}' for v in customer_counts], textposition='inside',
        hovertemplate='%{x}<br>Customer: %{y:,}<extra></extra>'
    ),
    row=1, col=2
)

# Total annotations
for i, ds in enumerate(datasets):
    total = labeler_counts[i] + customer_counts[i]
    fig_plotly.add_annotation(
        x=ds, y=total + 1200,
        text=f"<b>{total:,}</b>", showarrow=False,
        font=dict(size=13, color="white"),
        row=1, col=2
    )

fig_plotly.update_layout(
    template="plotly_dark",
    title=dict(text="InstructGPT: Task Distribution & Dataset Composition", font=dict(size=18)),
    barmode='stack',
    margin=dict(t=80, b=60, l=50, r=30),
    legend=dict(x=0.7, y=0.15, bgcolor="rgba(22,27,34,0.8)"),
)
fig_plotly.update_yaxes(title_text="Number of Prompts", row=1, col=2)

save_plotly_html(fig_plotly, OUTPUT_DIR_INTERACTIVE / "04_task_distribution_dataset.html")
