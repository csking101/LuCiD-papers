"""
Visualization 4: Bradley-Terry Model
Interactive visualization of the preference prediction model from Equation 1.

P[sigma1 > sigma2] = exp(sum r_hat(sigma1)) / (exp(sum r_hat(sigma1)) + exp(sum r_hat(sigma2)))

This is equivalent to a sigmoid over the reward difference:
P[sigma1 > sigma2] = sigmoid(R1 - R2)

where R1 = sum r_hat over sigma1, R2 = sum r_hat over sigma2.

Produces:
- Interactive Plotly HTML with sliders (CDN-based, ~50KB)
- Static Matplotlib PNG
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bt_with_noise(x, epsilon=0.1):
    """Bradley-Terry with human error probability epsilon.
    P = epsilon * 0.5 + (1 - epsilon) * sigmoid(x)
    Paper uses epsilon = 0.1 (10% random response chance).
    """
    return epsilon * 0.5 + (1 - epsilon) * sigmoid(x)


def plot_static():
    """Static PNG: Bradley-Terry sigmoid + annotated version with noise."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Bradley-Terry Preference Model',
        fontsize=16, fontweight='bold', y=1.02
    )

    x = np.linspace(-8, 8, 500)

    # Left: Pure Bradley-Terry sigmoid
    y = sigmoid(x)
    ax1.plot(x, y, color='#2E86C1', linewidth=2.5)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.fill_between(x, y, 0.5, where=(x > 0), alpha=0.15, color='#27AE60', label='Prefer segment 1')
    ax1.fill_between(x, y, 0.5, where=(x < 0), alpha=0.15, color='#E74C3C', label='Prefer segment 2')

    # Annotations
    ax1.annotate('R₁ >> R₂\nStrongly prefer σ¹',
                 xy=(5, sigmoid(5)), xytext=(3, 0.7),
                 arrowprops=dict(arrowstyle='->', color='#27AE60'),
                 fontsize=9, color='#27AE60', fontweight='bold')
    ax1.annotate('R₁ << R₂\nStrongly prefer σ²',
                 xy=(-5, sigmoid(-5)), xytext=(-7, 0.3),
                 arrowprops=dict(arrowstyle='->', color='#E74C3C'),
                 fontsize=9, color='#E74C3C', fontweight='bold')
    ax1.annotate('R₁ ≈ R₂\nUncertain',
                 xy=(0, 0.5), xytext=(2, 0.35),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=9, color='gray')

    ax1.set_xlabel('Reward Difference (R₁ - R₂)', fontsize=12)
    ax1.set_ylabel('P[σ¹ ≻ σ²]', fontsize=12)
    ax1.set_title('Pure Bradley-Terry Model', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(-0.05, 1.05)

    # Right: With human noise (epsilon = 0.1)
    for eps, color, label in [(0.0, '#2E86C1', 'ε = 0 (perfect)'),
                               (0.1, '#E8790C', 'ε = 0.1 (paper default)'),
                               (0.2, '#8B45A6', 'ε = 0.2 (noisy)'),
                               (0.5, '#E74C3C', 'ε = 0.5 (random)')]:
        y = bt_with_noise(x, epsilon=eps)
        ax2.plot(x, y, color=color, linewidth=2, label=label, alpha=0.9)

    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Show the bounded range
    ax2.axhline(0.95, color='#E8790C', linestyle=':', alpha=0.3)
    ax2.axhline(0.05, color='#E8790C', linestyle=':', alpha=0.3)
    ax2.annotate('Upper bound: 1 - ε/2 = 0.95',
                 xy=(6, 0.95), fontsize=8, color='#E8790C', alpha=0.7)
    ax2.annotate('Lower bound: ε/2 = 0.05',
                 xy=(3, 0.05), fontsize=8, color='#E8790C', alpha=0.7)

    ax2.set_xlabel('Reward Difference (R₁ - R₂)', fontsize=12)
    ax2.set_ylabel('P[σ¹ ≻ σ²]', fontsize=12)
    ax2.set_title('With Human Error Probability ε', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_ylim(-0.05, 1.05)

    # Add equation text box
    eq_text = (
        r'$\hat{P}[\sigma^1 \succ \sigma^2] = '
        r'\frac{e^{\sum \hat{r}(\sigma^1)}}{e^{\sum \hat{r}(\sigma^1)} + e^{\sum \hat{r}(\sigma^2)}}$'
    )
    fig.text(0.5, -0.04, eq_text, ha='center', fontsize=14,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8F5', edgecolor='#2E86C1', alpha=0.8))

    plt.tight_layout()
    output_path = OUTPUT_DIR_STATIC / "04_bradley_terry.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved: {output_path}")


def plot_interactive():
    """Interactive Plotly: slider for epsilon, hover to see exact values."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Bradley-Terry Preference Probability',
            'Effect of Human Error Rate (ε)'
        ),
        horizontal_spacing=0.1
    )

    x = np.linspace(-8, 8, 300)

    # Left panel: pure sigmoid with annotations
    y = sigmoid(x)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',
        name='P[σ¹ ≻ σ²]',
        line=dict(color='#2E86C1', width=3),
        hovertemplate='Reward diff: %{x:.2f}<br>P(prefer σ¹): %{y:.3f}<extra></extra>'
    ), row=1, col=1)

    fig.add_hline(y=0.5, line_dash='dash', line_color='gray', opacity=0.5, row=1, col=1)
    fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5, row=1, col=1)

    # Right panel: multiple epsilon values
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    colors = ['#2E86C1', '#1ABC9C', '#E8790C', '#F39C12', '#8B45A6', '#E74C3C', '#95A5A6']

    for eps, color in zip(epsilons, colors):
        y_eps = bt_with_noise(x, epsilon=eps)
        visible = eps in [0.0, 0.1, 0.2, 0.5]
        fig.add_trace(go.Scatter(
            x=x, y=y_eps, mode='lines',
            name=f'ε = {eps}',
            line=dict(color=color, width=2.5 if eps == 0.1 else 1.5,
                      dash='solid' if eps == 0.1 else 'dot' if eps == 0.0 else 'solid'),
            visible=True if visible else 'legendonly',
            hovertemplate=f'ε={eps}<br>Reward diff: %{{x:.2f}}<br>P(prefer σ¹): %{{y:.3f}}<extra></extra>'
        ), row=1, col=2)

    fig.add_hline(y=0.5, line_dash='dash', line_color='gray', opacity=0.3, row=1, col=2)
    fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.3, row=1, col=2)

    fig.update_layout(
        title=dict(
            text=('Bradley-Terry Preference Model<br>'
                  '<sub>P[σ¹ ≻ σ²] = exp(ΣR₁) / (exp(ΣR₁) + exp(ΣR₂)) — '
                  'with ε probability of random human error</sub>'),
            x=0.5, font=dict(size=18)
        ),
        width=1200, height=550,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        legend=dict(x=0.85, y=0.3, bgcolor='rgba(255,255,255,0.8)'),
        xaxis=dict(title='Reward Difference (R₁ - R₂)'),
        yaxis=dict(title='P[σ¹ ≻ σ²]', range=[-0.05, 1.05]),
        xaxis2=dict(title='Reward Difference (R₁ - R₂)'),
        yaxis2=dict(title='P[σ¹ ≻ σ²]', range=[-0.05, 1.05]),
    )

    # Add annotations
    fig.add_annotation(
        x=5, y=sigmoid(5), text='Strong preference<br>for σ¹',
        showarrow=True, arrowhead=2, arrowcolor='#27AE60',
        font=dict(color='#27AE60', size=11), row=1, col=1
    )
    fig.add_annotation(
        x=-5, y=sigmoid(-5), text='Strong preference<br>for σ²',
        showarrow=True, arrowhead=2, arrowcolor='#E74C3C',
        font=dict(color='#E74C3C', size=11), row=1, col=1
    )

    output_path = OUTPUT_DIR_INTERACTIVE / "04_bradley_terry.html"
    save_plotly_html(fig, output_path)


if __name__ == "__main__":
    plot_static()
    plot_interactive()
