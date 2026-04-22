"""
Visualization 9: Ablation Study Heatmap
Summarizes the ablation studies from Section 3.3 of "Deep RL from Human Preferences"

Shows relative performance of different ablation variants across tasks:
- Random queries (no disagreement-based selection)
- No ensemble (single predictor)
- No online queries (offline only)
- No regularization (dropout only, no L2)
- No segments (single frames, MuJoCo only)
- Target (regression to true reward instead of comparisons)

Produces: Static heatmap (Matplotlib) + Interactive heatmap (Plotly HTML, CDN-based)
"""

import sys
from pathlib import Path

# Add repo root for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from shared.style import apply_style
from shared.plotly_utils import save_plotly_html

apply_style()

OUTPUT_DIR_STATIC = Path(__file__).parent.parent / "output" / "static"
OUTPUT_DIR_INTERACTIVE = Path(__file__).parent.parent / "output" / "interactive"

# --- Ablation data (approximate relative performance from Figures 5 & 6) ---

mujoco_ablations = {
    'tasks': ['Reacher', 'Half Cheetah', 'Hopper', 'Walker', 'Swimmer', 'Ant', 'Pendulum', 'Dbl Pendulum'],
    'variants': ['Full Method', 'Random Queries', 'No Ensemble', 'Offline Only', 'No Regularization', 'No Segments', 'Target (Regression)'],
    'data': np.array([
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # Full
        [0.95, 0.90, 0.85, 0.92, 0.88, 0.80, 0.95, 0.90],  # Random queries
        [0.85, 0.80, 0.70, 0.75, 0.82, 0.65, 0.90, 0.80],  # No ensemble
        [0.60, 0.40, 0.30, 0.35, 0.50, 0.25, 0.70, 0.45],  # Offline only
        [0.80, 0.75, 0.65, 0.70, 0.78, 0.60, 0.85, 0.72],  # No regularization
        [0.50, 0.55, 0.40, 0.45, 0.60, 0.35, 0.65, 0.50],  # No segments
        [0.70, 0.60, 0.55, 0.50, 0.65, 0.45, 0.80, 0.60],  # Target
    ])
}

atari_ablations = {
    'tasks': ['Beamrider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'Space Inv.'],
    'variants': ['Full Method', 'Random Queries', 'No Ensemble', 'Offline Only', 'No Regularization', 'Target (Regression)'],
    'data': np.array([
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # Full
        [0.90, 0.95, 0.85, 0.98, 0.80, 0.88, 0.92],  # Random queries
        [0.75, 0.80, 0.70, 0.90, 0.60, 0.72, 0.78],  # No ensemble
        [0.35, 0.25, 0.20, 0.50, 0.15, 0.30, 0.40],  # Offline only
        [0.70, 0.72, 0.65, 0.85, 0.55, 0.68, 0.70],  # No regularization
        [0.80, 0.85, 0.75, 0.95, 0.70, 0.78, 0.82],  # Target
    ])
}


def plot_static_heatmap():
    """Matplotlib static heatmap for both domains."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), gridspec_kw={'width_ratios': [8, 7]})
    fig.suptitle(
        'Ablation Study — Relative Performance of Algorithm Components',
        fontsize=16, fontweight='bold', y=1.02
    )
    fig.text(
        0.5, 0.97,
        'Values show fraction of full method performance (1.0 = full method). Lower = more important component.',
        ha='center', fontsize=11, color='#555555'
    )

    cmap = plt.cm.RdYlGn

    # MuJoCo heatmap
    im1 = ax1.imshow(mujoco_ablations['data'], cmap=cmap, vmin=0, vmax=1.05, aspect='auto')
    ax1.set_xticks(range(len(mujoco_ablations['tasks'])))
    ax1.set_xticklabels(mujoco_ablations['tasks'], rotation=45, ha='right', fontsize=9)
    ax1.set_yticks(range(len(mujoco_ablations['variants'])))
    ax1.set_yticklabels(mujoco_ablations['variants'], fontsize=9)
    ax1.set_title('MuJoCo Tasks', fontsize=13, fontweight='bold', pad=10)

    for i in range(len(mujoco_ablations['variants'])):
        for j in range(len(mujoco_ablations['tasks'])):
            val = mujoco_ablations['data'][i, j]
            color = 'white' if val < 0.5 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

    # Atari heatmap
    im2 = ax2.imshow(atari_ablations['data'], cmap=cmap, vmin=0, vmax=1.05, aspect='auto')
    ax2.set_xticks(range(len(atari_ablations['tasks'])))
    ax2.set_xticklabels(atari_ablations['tasks'], rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(range(len(atari_ablations['variants'])))
    ax2.set_yticklabels(atari_ablations['variants'], fontsize=9)
    ax2.set_title('Atari Tasks', fontsize=13, fontweight='bold', pad=10)

    for i in range(len(atari_ablations['variants'])):
        for j in range(len(atari_ablations['tasks'])):
            val = atari_ablations['data'][i, j]
            color = 'white' if val < 0.5 else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', fraction=0.04, pad=0.18)
    cbar.set_label('Relative Performance (fraction of full method)', fontsize=10)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    output_path = OUTPUT_DIR_STATIC / "09_ablation_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved: {output_path}")


def plot_interactive_heatmap():
    """Plotly interactive heatmap with hover details."""
    # Combine both domains into one view
    all_tasks = mujoco_ablations['tasks'] + ['  |  '] + atari_ablations['tasks']

    # Shared variants (pad Atari with NaN for "No Segments")
    all_variants = ['Full Method', 'Random Queries', 'No Ensemble', 'Offline Only',
                    'No Regularization', 'No Segments', 'Target (Regression)']

    # Build combined matrix
    n_variants = len(all_variants)
    n_tasks = len(all_tasks)
    combined = np.full((n_variants, n_tasks), np.nan)

    # Fill MuJoCo
    for i, var in enumerate(mujoco_ablations['variants']):
        row_idx = all_variants.index(var)
        for j in range(len(mujoco_ablations['tasks'])):
            combined[row_idx, j] = mujoco_ablations['data'][i, j]

    # Fill Atari (offset by len(mujoco_tasks) + 1 for separator)
    offset = len(mujoco_ablations['tasks']) + 1
    for i, var in enumerate(atari_ablations['variants']):
        row_idx = all_variants.index(var)
        for j in range(len(atari_ablations['tasks'])):
            combined[row_idx, offset + j] = atari_ablations['data'][i, j]

    # Custom hover text
    hover_text = []
    for i, var in enumerate(all_variants):
        row = []
        for j, task in enumerate(all_tasks):
            val = combined[i, j]
            if np.isnan(val):
                row.append('')
            else:
                domain = 'MuJoCo' if j < len(mujoco_ablations['tasks']) else 'Atari'
                impact = 'Critical' if val < 0.4 else 'Important' if val < 0.7 else 'Moderate' if val < 0.9 else 'Minor'
                row.append(f'<b>{task}</b><br>Variant: {var}<br>Domain: {domain}<br>Rel. Perf: {val:.2f}<br>Impact: {impact}')
        hover_text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=combined,
        x=all_tasks,
        y=all_variants,
        hovertext=hover_text,
        hoverinfo='text',
        colorscale='RdYlGn',
        zmin=0,
        zmax=1.05,
        colorbar=dict(title=dict(text='Relative<br>Performance', side='right')),
        text=[[f'{v:.2f}' if not np.isnan(v) else '' for v in row] for row in combined],
        texttemplate='%{text}',
        textfont=dict(size=11),
    ))

    fig.update_layout(
        title=dict(
            text='Ablation Study: Impact of Each Algorithm Component<br><sub>Values show fraction of full method performance. Lower = more critical component.</sub>',
            x=0.5,
            font=dict(size=18),
        ),
        xaxis=dict(title='Tasks', tickangle=-45, side='bottom'),
        yaxis=dict(title='Ablation Variant', autorange='reversed'),
        width=1100,
        height=500,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        margin=dict(l=180, r=50, t=100, b=120),
    )

    # Add domain separator annotation
    fig.add_vline(x=7.5, line_width=3, line_dash='dash', line_color='gray')
    fig.add_annotation(x=3.5, y=-0.15, text='MuJoCo', showarrow=False,
                       font=dict(size=14, color='#2E86C1'), xref='x', yref='paper')
    fig.add_annotation(x=12, y=-0.15, text='Atari', showarrow=False,
                       font=dict(size=14, color='#2E86C1'), xref='x', yref='paper')

    output_path = OUTPUT_DIR_INTERACTIVE / "09_ablation_heatmap.html"
    save_plotly_html(fig, output_path)


if __name__ == "__main__":
    plot_static_heatmap()
    plot_interactive_heatmap()
