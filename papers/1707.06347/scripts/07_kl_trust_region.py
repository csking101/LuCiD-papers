# 07 — KL Divergence Trust Region
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# 2D parameter space with contours of the surrogate objective.
# Trust region shown as a KL-divergence ball around theta_old.
# Interactive: slider for delta to resize the trust region.
#
# Run:
#   python 07_kl_trust_region.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go

from shared.style import apply_style, COLORS
from shared.plotly_utils import save_plotly_html

apply_style()

OUTPUT_DIR_STATIC = Path(__file__).parent.parent / "output" / "static"
OUTPUT_DIR_INTERACTIVE = Path(__file__).parent.parent / "output" / "interactive"
OUTPUT_DIR_STATIC.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_INTERACTIVE.mkdir(parents=True, exist_ok=True)


def surrogate_objective(theta1, theta2):
    """Simulated surrogate objective landscape."""
    return -(0.5 * (theta1 - 1)**2 + 0.3 * (theta2 - 0.5)**2) + 2


def plot_static():
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create grid
    t1 = np.linspace(-2, 3, 200)
    t2 = np.linspace(-2, 3, 200)
    T1, T2 = np.meshgrid(t1, t2)
    Z = surrogate_objective(T1, T2)

    # Contour plot
    contour = ax.contourf(T1, T2, Z, levels=30, cmap='viridis', alpha=0.8)
    ax.contour(T1, T2, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax, label='Surrogate Objective L(theta)')

    # theta_old
    theta_old = (-0.5, -0.5)
    ax.plot(*theta_old, 'o', color=COLORS['blue'], markersize=12, zorder=5)
    ax.annotate('theta_old', theta_old, textcoords="offset points",
                xytext=(10, 10), fontsize=12, color=COLORS['blue'], fontweight='bold')

    # Trust region (KL ball)
    delta = 0.8
    circle = patches.Circle(theta_old, delta, linewidth=2.5, edgecolor=COLORS['purple'],
                             facecolor=COLORS['purple'], alpha=0.15, linestyle='--')
    ax.add_patch(circle)
    ax.annotate(f'Trust Region (delta={delta})', (theta_old[0] + delta * 0.5, theta_old[1] + delta + 0.1),
                fontsize=11, color=COLORS['purple'], fontweight='bold')

    # Unconstrained optimum (global)
    global_opt = (1.0, 0.5)
    ax.plot(*global_opt, '*', color=COLORS['red'], markersize=15, zorder=5)
    ax.annotate('Unconstrained\noptimum', global_opt, textcoords="offset points",
                xytext=(15, -20), fontsize=10, color=COLORS['red'])

    # TRPO solution (best point inside trust region)
    # Project global optimum onto trust region boundary
    direction = np.array(global_opt) - np.array(theta_old)
    direction = direction / np.linalg.norm(direction)
    trpo_sol = np.array(theta_old) + delta * direction
    ax.plot(*trpo_sol, 'D', color=COLORS['green'], markersize=10, zorder=5)
    ax.annotate('TRPO solution\n(best in region)', trpo_sol, textcoords="offset points",
                xytext=(15, 10), fontsize=10, color=COLORS['green'], fontweight='bold')

    # Arrow from old to TRPO solution
    ax.annotate('', xy=trpo_sol, xytext=theta_old,
                arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2))

    ax.set_xlabel('theta_1', fontsize=12)
    ax.set_ylabel('theta_2', fontsize=12)
    ax.set_title('Trust Region in Parameter Space', fontsize=14, fontweight='bold')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')

    plt.tight_layout()
    output_path = OUTPUT_DIR_STATIC / "07_kl_trust_region.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved: {output_path}")


def plot_interactive():
    # Create grid
    t1 = np.linspace(-2, 3, 100)
    t2 = np.linspace(-2, 3, 100)
    T1, T2 = np.meshgrid(t1, t2)
    Z = surrogate_objective(T1, T2)

    theta_old = (-0.5, -0.5)
    global_opt = (1.0, 0.5)

    fig = go.Figure()

    # Background contour (always visible)
    fig.add_trace(go.Contour(
        x=t1, y=t2, z=Z,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="L(theta)"),
        contours=dict(showlines=True, coloring='heatmap'),
        name='Surrogate',
    ))

    # theta_old point
    fig.add_trace(go.Scatter(
        x=[theta_old[0]], y=[theta_old[1]],
        mode='markers+text', marker=dict(size=14, color='#2E86C1', symbol='circle'),
        text=['theta_old'], textposition='top right',
        textfont=dict(color='#2E86C1', size=13),
        name='theta_old',
    ))

    # Global optimum
    fig.add_trace(go.Scatter(
        x=[global_opt[0]], y=[global_opt[1]],
        mode='markers+text', marker=dict(size=14, color='#E74C3C', symbol='star'),
        text=['Unconstrained optimum'], textposition='bottom right',
        textfont=dict(color='#E74C3C', size=11),
        name='Global opt',
    ))

    # Trust regions for different deltas
    deltas = np.linspace(0.2, 2.0, 19)
    theta_arr = np.linspace(0, 2 * np.pi, 80)

    for i, delta in enumerate(deltas):
        circle_x = theta_old[0] + delta * np.cos(theta_arr)
        circle_y = theta_old[1] + delta * np.sin(theta_arr)

        # TRPO solution
        direction = np.array(global_opt) - np.array(theta_old)
        dist = np.linalg.norm(direction)
        direction_norm = direction / dist
        if dist <= delta:
            trpo = np.array(global_opt)
        else:
            trpo = np.array(theta_old) + delta * direction_norm

        visible = (i == 6)  # Default delta ~ 0.8

        fig.add_trace(go.Scatter(
            x=circle_x.tolist(), y=circle_y.tolist(),
            mode='lines', line=dict(color='#8B45A6', width=2, dash='dash'),
            fill='toself', fillcolor='rgba(139, 69, 166, 0.1)',
            name=f'Trust Region (delta={delta:.1f})',
            showlegend=False, visible=visible,
        ))

        fig.add_trace(go.Scatter(
            x=[trpo[0]], y=[trpo[1]],
            mode='markers+text', marker=dict(size=12, color='#27AE60', symbol='diamond'),
            text=['TRPO solution'], textposition='top right',
            textfont=dict(color='#27AE60', size=11),
            name='TRPO', showlegend=False, visible=visible,
        ))

    # Slider
    steps = []
    for i, delta in enumerate(deltas):
        step = dict(
            method="update",
            args=[{"visible": [True, True, True] + [False] * (len(deltas) * 2)}],
            label=f"{delta:.1f}"
        )
        step["args"][0]["visible"][3 + i * 2] = True
        step["args"][0]["visible"][3 + i * 2 + 1] = True
        steps.append(step)

    fig.update_layout(
        sliders=[dict(
            active=6,
            currentvalue={"prefix": "Trust Region delta = "},
            pad={"t": 50},
            steps=steps,
        )],
        title="KL Divergence Trust Region",
        template="plotly_dark",
        height=600,
        autosize=True,
        xaxis_title="theta_1",
        yaxis_title="theta_2",
        xaxis=dict(scaleanchor="y"),
        margin=dict(t=80, b=100, l=60, r=60),
    )

    output_path = OUTPUT_DIR_INTERACTIVE / "07_kl_trust_region.html"
    save_plotly_html(fig, output_path)


if __name__ == "__main__":
    plot_static()
    plot_interactive()
