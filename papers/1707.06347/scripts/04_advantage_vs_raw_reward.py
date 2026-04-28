# 04 — Advantage vs Raw Reward
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Shows why the advantage function reduces variance compared to raw reward.
# Two panels: gradient signal using raw reward R(tau) vs advantage A = Q - V.
# Interactive version lets you adjust the baseline.
#
# Run:
#   python 04_advantage_vs_raw_reward.py

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

np.random.seed(42)


def generate_data():
    """Simulate gradient signals with raw reward vs advantage."""
    n_actions = 50

    # True Q values for actions (some good, some bad)
    q_values = np.random.normal(50, 15, n_actions)
    v_baseline = np.mean(q_values)  # V(s) is roughly the average

    # Raw reward signal: always positive, high variance
    raw_signal = q_values * np.random.normal(1, 0.3, n_actions)

    # Advantage signal: centered around zero, lower variance
    advantage_signal = (q_values - v_baseline) * np.random.normal(1, 0.3, n_actions)

    return q_values, raw_signal, advantage_signal, v_baseline


def plot_static():
    q_values, raw_signal, advantage_signal, v_baseline = generate_data()
    n = len(raw_signal)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Raw Reward
    colors1 = [COLORS['green'] if r > np.mean(raw_signal) else COLORS['blue'] for r in raw_signal]
    ax1.bar(range(n), raw_signal, color=colors1, alpha=0.7, width=0.8)
    ax1.axhline(y=np.mean(raw_signal), color=COLORS['red'], linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(raw_signal):.1f}')
    ax1.set_title('Gradient Signal: Raw Reward R(tau)', fontsize=13)
    ax1.set_xlabel('Action Index')
    ax1.set_ylabel('Gradient Signal (reward * grad log pi)')
    ax1.legend(fontsize=9)
    ax1.set_ylim(bottom=0)
    # Annotate variance
    ax1.text(0.02, 0.95, f'Variance: {np.var(raw_signal):.0f}',
             transform=ax1.transAxes, fontsize=11, color=COLORS['red'],
             fontweight='bold', va='top')
    ax1.text(0.02, 0.88, 'All positive -> everything reinforced',
             transform=ax1.transAxes, fontsize=9, color=COLORS['grey'], va='top')

    # Panel 2: Advantage
    colors2 = [COLORS['green'] if a > 0 else COLORS['red'] for a in advantage_signal]
    ax2.bar(range(n), advantage_signal, color=colors2, alpha=0.7, width=0.8)
    ax2.axhline(y=0, color=COLORS['grey'], linestyle='-', linewidth=1)
    ax2.set_title('Gradient Signal: Advantage A = Q - V', fontsize=13)
    ax2.set_xlabel('Action Index')
    ax2.set_ylabel('Gradient Signal (advantage * grad log pi)')
    # Annotate variance
    ax2.text(0.02, 0.95, f'Variance: {np.var(advantage_signal):.0f}',
             transform=ax2.transAxes, fontsize=11, color=COLORS['green'],
             fontweight='bold', va='top')
    ax2.text(0.02, 0.88, 'Centered: good up, bad down',
             transform=ax2.transAxes, fontsize=9, color=COLORS['grey'], va='top')

    plt.tight_layout()
    output_path = OUTPUT_DIR_STATIC / "04_advantage_vs_raw_reward.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved: {output_path}")


def plot_interactive():
    q_values, _, _, v_baseline = generate_data()
    n = len(q_values)
    grad_noise = np.random.normal(1, 0.3, n)

    # Create figure with slider for baseline
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Raw Reward (no baseline)",
                                        "With Baseline Subtracted"])

    # Precompute for multiple baseline values
    baselines = np.linspace(0, 80, 41)

    for i, bl in enumerate(baselines):
        raw = q_values * grad_noise
        adv = (q_values - bl) * grad_noise
        colors_raw = ['#27AE60' if r > np.mean(raw) else '#2E86C1' for r in raw]
        colors_adv = ['#27AE60' if a > 0 else '#E74C3C' for a in adv]

        visible = (i == 20)  # Default: baseline = mean

        fig.add_trace(go.Bar(x=list(range(n)), y=raw, marker_color=colors_raw,
                              name='Raw', showlegend=False, visible=visible),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=list(range(n)), y=adv.tolist(), marker_color=colors_adv,
                              name='Advantage', showlegend=False, visible=visible),
                      row=1, col=2)

    # Create slider steps
    steps = []
    for i, bl in enumerate(baselines):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=f"{bl:.0f}"
        )
        step["args"][0]["visible"][i * 2] = True
        step["args"][0]["visible"][i * 2 + 1] = True
        steps.append(step)

    fig.update_layout(
        sliders=[dict(
            active=20,
            currentvalue={"prefix": "Baseline V(s) = "},
            pad={"t": 50},
            steps=steps,
        )],
        title="Advantage vs Raw Reward: Effect of Baseline",
        template="plotly_dark",
        height=500,
        margin=dict(t=80, b=100),
    )

    output_path = OUTPUT_DIR_INTERACTIVE / "04_advantage_vs_raw_reward.html"
    save_plotly_html(fig, output_path)


if __name__ == "__main__":
    plot_static()
    plot_interactive()
