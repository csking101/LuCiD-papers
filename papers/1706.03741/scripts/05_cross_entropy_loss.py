"""
Visualization 5: Cross-Entropy Loss Surface
Interactive 3D visualization of the loss function used to train the reward model.

loss(r_hat) = -sum[ mu(1) * log P[sigma1 > sigma2] + mu(2) * log P[sigma2 > sigma1] ]

For a single comparison, this simplifies to:
loss = -[ mu * log(p) + (1-mu) * log(1-p) ]

where:
- p = predicted probability that sigma1 is preferred
- mu = true label (1 if sigma1 preferred, 0 if sigma2, 0.5 if tie)

Produces:
- Interactive 3D Plotly surface (CDN-based, ~50KB)
- Static 2D Matplotlib showing loss curves for different label types
"""

import sys
from pathlib import Path

# Add repo root for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.style import apply_style
from shared.plotly_utils import save_plotly_html

apply_style()

OUTPUT_DIR_STATIC = Path(__file__).parent.parent / "output" / "static"
OUTPUT_DIR_INTERACTIVE = Path(__file__).parent.parent / "output" / "interactive"


def cross_entropy(p, mu):
    """Cross-entropy loss for a single comparison.
    p: predicted probability of preferring sigma1
    mu: true label (mu=1 means sigma1 preferred, mu=0 means sigma2 preferred)
    """
    eps = 1e-7  # numerical stability
    p = np.clip(p, eps, 1 - eps)
    return -(mu * np.log(p) + (1 - mu) * np.log(1 - p))


def plot_static():
    """Static 2D: loss curves for different label types."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Cross-Entropy Loss for Reward Model Training',
        fontsize=16, fontweight='bold', y=1.02
    )

    p = np.linspace(0.01, 0.99, 500)

    # Left: Loss curves for different true labels
    labels = [
        (1.0, 'σ¹ preferred (μ=1)', '#27AE60'),
        (0.0, 'σ² preferred (μ=0)', '#E74C3C'),
        (0.5, 'Equally good (μ=0.5)', '#2E86C1'),
    ]

    for mu, label, color in labels:
        loss = cross_entropy(p, mu)
        ax1.plot(p, loss, color=color, linewidth=2.5, label=label)

        # Mark minimum
        min_idx = np.argmin(loss)
        ax1.plot(p[min_idx], loss[min_idx], 'o', color=color, markersize=8, zorder=5)

    ax1.set_xlabel('Predicted P[σ¹ ≻ σ²]', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Loss vs Predicted Probability', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 4)

    # Annotations
    ax1.annotate('Minimum at p=1\n(correct & confident)',
                 xy=(0.99, cross_entropy(0.99, 1.0)), xytext=(0.6, 2.5),
                 arrowprops=dict(arrowstyle='->', color='#27AE60'),
                 fontsize=9, color='#27AE60')
    ax1.annotate('Minimum at p=0.5\n(uncertain → correct)',
                 xy=(0.5, cross_entropy(0.5, 0.5)), xytext=(0.15, 2.0),
                 arrowprops=dict(arrowstyle='->', color='#2E86C1'),
                 fontsize=9, color='#2E86C1')

    # Right: Effect of reward difference on loss
    reward_diff = np.linspace(-6, 6, 500)
    p_pred = 1 / (1 + np.exp(-reward_diff))  # sigmoid

    for mu, label, color in labels:
        loss = cross_entropy(p_pred, mu)
        ax2.plot(reward_diff, loss, color=color, linewidth=2.5, label=label)

    ax2.set_xlabel('Reward Difference (R₁ - R₂)', fontsize=12)
    ax2.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax2.set_title('Loss vs Reward Difference', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 7)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.4)

    # Annotations
    ax2.annotate('High R₁, σ¹ preferred\n→ low loss (correct)',
                 xy=(4, cross_entropy(1 / (1 + np.exp(-4)), 1.0)),
                 xytext=(1, 4),
                 arrowprops=dict(arrowstyle='->', color='#27AE60'),
                 fontsize=9, color='#27AE60')
    ax2.annotate('High R₁, but σ² preferred\n→ high loss (wrong!)',
                 xy=(4, cross_entropy(1 / (1 + np.exp(-4)), 0.0)),
                 xytext=(0.5, 5.5),
                 arrowprops=dict(arrowstyle='->', color='#E74C3C'),
                 fontsize=9, color='#E74C3C')

    # Equation box
    eq_text = (
        r'$\mathcal{L}(\hat{r}) = -\sum \left[ \mu(1) \log \hat{P}[\sigma^1 \succ \sigma^2]'
        r' + \mu(2) \log \hat{P}[\sigma^2 \succ \sigma^1] \right]$'
    )
    fig.text(0.5, -0.04, eq_text, ha='center', fontsize=13,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEDEC', edgecolor='#E74C3C', alpha=0.8))

    plt.tight_layout()
    output_path = OUTPUT_DIR_STATIC / "05_cross_entropy_loss.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved: {output_path}")


def plot_interactive():
    """Interactive 3D surface: loss as function of p and mu."""
    # Create mesh
    p_vals = np.linspace(0.01, 0.99, 100)
    mu_vals = np.linspace(0, 1, 100)
    P, MU = np.meshgrid(p_vals, mu_vals)
    Z = cross_entropy(P, MU)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'xy'}]],
        subplot_titles=('3D Loss Surface', 'Loss Contours'),
        horizontal_spacing=0.08
    )

    # Left: 3D surface
    fig.add_trace(go.Surface(
        x=p_vals, y=mu_vals, z=Z,
        colorscale='Viridis',
        colorbar=dict(title=dict(text='Loss'), x=0.45, len=0.8),
        hovertemplate='P(prefer σ¹): %{x:.2f}<br>True label μ: %{y:.2f}<br>Loss: %{z:.3f}<extra></extra>',
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor='limegreen', project_z=True)
        )
    ), row=1, col=1)

    # Right: 2D contour plot
    fig.add_trace(go.Contour(
        x=p_vals, y=mu_vals, z=Z,
        colorscale='Viridis',
        colorbar=dict(title=dict(text='Loss'), x=1.02, len=0.8),
        contours=dict(coloring='heatmap', showlabels=True,
                      labelfont=dict(size=10, color='white')),
        hovertemplate='P(prefer σ¹): %{x:.2f}<br>True label μ: %{y:.2f}<br>Loss: %{z:.3f}<extra></extra>'
    ), row=1, col=2)

    # Add diagonal line (optimal: p = mu)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Optimal (p = μ)',
        showlegend=True,
        hoverinfo='skip'
    ), row=1, col=2)

    # Add key points
    key_points = [
        (1.0, 1.0, 'Correct: σ¹ preferred, p≈1'),
        (0.0, 0.0, 'Correct: σ² preferred, p≈0'),
        (0.5, 0.5, 'Tie: equal preference'),
        (0.0, 1.0, 'Wrong: σ¹ preferred, p≈0'),
        (1.0, 0.0, 'Wrong: σ² preferred, p≈1'),
    ]
    for px, my, txt in key_points:
        px_c = np.clip(px, 0.02, 0.98)
        fig.add_trace(go.Scatter(
            x=[px_c], y=[my],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='circle'),
            text=[txt], textposition='top center',
            textfont=dict(size=9),
            showlegend=False,
            hoverinfo='text',
            hovertext=txt
        ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text=('Cross-Entropy Loss for Reward Model<br>'
                  '<sub>L = -[μ·log(p) + (1-μ)·log(1-p)] where p = predicted preference, μ = true label</sub>'),
            x=0.5, font=dict(size=17)
        ),
        width=1200, height=600,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        scene=dict(
            xaxis_title='P(prefer σ¹)',
            yaxis_title='True label μ',
            zaxis_title='Loss',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        ),
        xaxis2=dict(title='P(prefer σ¹)'),
        yaxis2=dict(title='True label μ'),
    )

    output_path = OUTPUT_DIR_INTERACTIVE / "05_cross_entropy_loss.html"
    save_plotly_html(fig, output_path)


if __name__ == "__main__":
    plot_static()
    plot_interactive()
