# 08 — Clipped Surrogate Objective
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Reproduces Figure 1 from the PPO paper — L^CLIP as a function of r_t
# for positive and negative advantage.
# Interactive: epsilon slider (0.1, 0.2, 0.3).
#
# Run:
#   python 08_clipped_surrogate.py

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


def compute_clip_objective(r, advantage, epsilon):
    """Compute L^CLIP for given r values."""
    unclipped = r * advantage
    clipped = np.clip(r, 1 - epsilon, 1 + epsilon) * advantage
    return np.minimum(unclipped, clipped)


def plot_static():
    epsilon = 0.2
    r = np.linspace(0.0, 2.0, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # === Positive Advantage ===
    A_pos = 1.0
    unclipped_pos = r * A_pos
    clipped_pos = np.clip(r, 1 - epsilon, 1 + epsilon) * A_pos
    L_clip_pos = np.minimum(unclipped_pos, clipped_pos)

    ax1.plot(r, unclipped_pos, '--', color=COLORS['grey'], alpha=0.5, linewidth=1.5, label='r * A (unclipped)')
    ax1.plot(r, L_clip_pos, color=COLORS['green'], linewidth=2.5, label='L^CLIP')
    ax1.axvline(x=1 - epsilon, color=COLORS['purple'], linestyle=':', alpha=0.7, label=f'1 - eps = {1-epsilon}')
    ax1.axvline(x=1 + epsilon, color=COLORS['purple'], linestyle=':', alpha=0.7, label=f'1 + eps = {1+epsilon}')
    ax1.axvline(x=1.0, color=COLORS['grey'], linestyle='-', alpha=0.3)
    ax1.set_title(f'Positive Advantage (A > 0), eps={epsilon}', fontsize=13, fontweight='bold')
    ax1.set_xlabel('r_t(theta) = pi_new / pi_old')
    ax1.set_ylabel('Objective')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 2)

    # Shade the flat region
    ax1.axhspan(ymin=(1 + epsilon) * A_pos - 0.01, ymax=(1 + epsilon) * A_pos + 0.01,
                xmin=0.6, xmax=1.0, alpha=0.1, color=COLORS['green'])

    # Annotate
    ax1.annotate('Clipped: no incentive\nto increase r further',
                 xy=(1.5, (1 + epsilon) * A_pos), xytext=(1.5, 0.5),
                 arrowprops=dict(arrowstyle='->', color=COLORS['orange']),
                 fontsize=9, color=COLORS['orange'])

    # === Negative Advantage ===
    A_neg = -1.0
    unclipped_neg = r * A_neg
    clipped_neg = np.clip(r, 1 - epsilon, 1 + epsilon) * A_neg
    L_clip_neg = np.minimum(unclipped_neg, clipped_neg)

    ax2.plot(r, unclipped_neg, '--', color=COLORS['grey'], alpha=0.5, linewidth=1.5, label='r * A (unclipped)')
    ax2.plot(r, L_clip_neg, color=COLORS['red'], linewidth=2.5, label='L^CLIP')
    ax2.axvline(x=1 - epsilon, color=COLORS['purple'], linestyle=':', alpha=0.7, label=f'1 - eps = {1-epsilon}')
    ax2.axvline(x=1 + epsilon, color=COLORS['purple'], linestyle=':', alpha=0.7, label=f'1 + eps = {1+epsilon}')
    ax2.axvline(x=1.0, color=COLORS['grey'], linestyle='-', alpha=0.3)
    ax2.set_title(f'Negative Advantage (A < 0), eps={epsilon}', fontsize=13, fontweight='bold')
    ax2.set_xlabel('r_t(theta) = pi_new / pi_old')
    ax2.set_ylabel('Objective')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 2)

    ax2.annotate('Clipped: no incentive\nto decrease r further',
                 xy=(0.5, (1 - epsilon) * A_neg), xytext=(0.3, -1.5),
                 arrowprops=dict(arrowstyle='->', color=COLORS['orange']),
                 fontsize=9, color=COLORS['orange'])

    plt.tight_layout()
    output_path = OUTPUT_DIR_STATIC / "08_clipped_surrogate.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved: {output_path}")


def plot_interactive():
    r = np.linspace(0.0, 2.0, 500)
    epsilons = np.arange(0.05, 0.55, 0.05)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Positive Advantage (A > 0)",
                                        "Negative Advantage (A < 0)"])

    for i, eps in enumerate(epsilons):
        visible = abs(eps - 0.2) < 0.01  # Default: eps=0.2

        # Positive advantage
        A_pos = 1.0
        L_pos = compute_clip_objective(r, A_pos, eps)
        unclipped_pos = r * A_pos

        fig.add_trace(go.Scatter(x=r, y=unclipped_pos, mode='lines',
                                  line=dict(color='grey', dash='dash', width=1),
                                  name='Unclipped', showlegend=False, visible=visible),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=r, y=L_pos, mode='lines',
                                  line=dict(color='#27AE60', width=3),
                                  name='L^CLIP', showlegend=False, visible=visible),
                      row=1, col=1)

        # Negative advantage
        A_neg = -1.0
        L_neg = compute_clip_objective(r, A_neg, eps)
        unclipped_neg = r * A_neg

        fig.add_trace(go.Scatter(x=r, y=unclipped_neg, mode='lines',
                                  line=dict(color='grey', dash='dash', width=1),
                                  name='Unclipped', showlegend=False, visible=visible),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=r, y=L_neg, mode='lines',
                                  line=dict(color='#E74C3C', width=3),
                                  name='L^CLIP', showlegend=False, visible=visible),
                      row=1, col=2)

    # Slider
    traces_per_eps = 4
    steps = []
    for i, eps in enumerate(epsilons):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=f"{eps:.2f}"
        )
        for j in range(traces_per_eps):
            step["args"][0]["visible"][i * traces_per_eps + j] = True
        steps.append(step)

    fig.update_layout(
        sliders=[dict(
            active=3,  # eps=0.2
            currentvalue={"prefix": "epsilon = "},
            pad={"t": 50},
            steps=steps,
        )],
        title="PPO Clipped Surrogate Objective (Figure 1 from paper)",
        template="plotly_dark",
        height=500,
        margin=dict(t=80, b=100),
    )

    fig.update_xaxes(title_text="r_t(theta)", row=1, col=1)
    fig.update_xaxes(title_text="r_t(theta)", row=1, col=2)
    fig.update_yaxes(title_text="Objective", row=1, col=1)
    fig.update_yaxes(title_text="Objective", row=1, col=2)

    output_path = OUTPUT_DIR_INTERACTIVE / "08_clipped_surrogate.html"
    save_plotly_html(fig, output_path)


if __name__ == "__main__":
    plot_static()
    plot_interactive()
