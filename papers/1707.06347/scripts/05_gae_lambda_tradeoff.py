# 05 — GAE Lambda Tradeoff
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# GAE advantage estimates for different lambda values (0, 0.5, 0.95, 1.0)
# on a sample trajectory. Shows bias-variance tradeoff.
# Interactive version has a lambda slider.
#
# Run:
#   python 05_gae_lambda_tradeoff.py

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

np.random.seed(42)

GAMMA = 0.99


def generate_trajectory(T=30):
    """Generate a sample trajectory with rewards and value estimates."""
    rewards = np.random.normal(1.0, 0.5, T)
    # Simulate a value function that's imperfect
    true_values = np.cumsum(rewards[::-1])[::-1] * 0.5
    value_estimates = true_values + np.random.normal(0, 0.3, T)
    # Pad with terminal value
    value_estimates = np.append(value_estimates, 0.0)
    return rewards, value_estimates


def compute_gae(rewards, values, gamma, lam):
    """Compute GAE advantage estimates."""
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


def plot_static():
    rewards, values = generate_trajectory(30)
    T = len(rewards)
    timesteps = np.arange(T)

    lambdas = [0.0, 0.5, 0.95, 1.0]
    colors_list = [COLORS['red'], COLORS['orange'], COLORS['green'], COLORS['blue']]
    labels = ['lambda=0 (TD, low var, high bias)',
              'lambda=0.5 (balanced)',
              'lambda=0.95 (PPO default)',
              'lambda=1 (MC, high var, no bias)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('GAE Advantage Estimates: Lambda Tradeoff', fontsize=14, fontweight='bold')

    for idx, (lam, color, label) in enumerate(zip(lambdas, colors_list, labels)):
        ax = axes[idx // 2][idx % 2]
        advantages = compute_gae(rewards, values, GAMMA, lam)

        ax.bar(timesteps, advantages, color=color, alpha=0.6, width=0.8)
        ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
        ax.set_title(label, fontsize=11, color=color)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Advantage')

        # Show variance
        ax.text(0.02, 0.95, f'Var: {np.var(advantages):.3f}',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=color, va='top')

    plt.tight_layout()
    output_path = OUTPUT_DIR_STATIC / "05_gae_lambda_tradeoff.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved: {output_path}")


def plot_interactive():
    rewards, values = generate_trajectory(30)
    T = len(rewards)
    timesteps = list(range(T))

    # Precompute for lambda slider
    lambda_values = np.linspace(0, 1, 51)

    fig = go.Figure()

    for i, lam in enumerate(lambda_values):
        advantages = compute_gae(rewards, values, GAMMA, lam)
        colors_bar = ['#27AE60' if a > 0 else '#E74C3C' for a in advantages]
        visible = (i == 48)  # Default: lambda = 0.95 (approx)

        fig.add_trace(go.Bar(
            x=timesteps, y=advantages.tolist(),
            marker_color=colors_bar,
            name=f'lambda={lam:.2f}',
            showlegend=False,
            visible=visible,
        ))

    steps = []
    for i, lam in enumerate(lambda_values):
        advantages = compute_gae(rewards, values, GAMMA, lam)
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"GAE Advantages (lambda={lam:.2f}, variance={np.var(advantages):.3f})"}],
            label=f"{lam:.2f}"
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    fig.update_layout(
        sliders=[dict(
            active=48,
            currentvalue={"prefix": "lambda = "},
            pad={"t": 50},
            steps=steps,
        )],
        title="GAE Advantages (lambda=0.95, PPO default)",
        template="plotly_dark",
        xaxis_title="Timestep",
        yaxis_title="Advantage Estimate",
        margin=dict(t=60, b=80, l=50, r=30),
    )

    output_path = OUTPUT_DIR_INTERACTIVE / "05_gae_lambda_tradeoff.html"
    save_plotly_html(fig, output_path)


if __name__ == "__main__":
    plot_static()
    plot_interactive()
