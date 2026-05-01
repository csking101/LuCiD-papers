# 06 — Reward Overoptimization
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# The key figure from the paper: predicted reward keeps climbing but true human
# preference peaks then drops. Demonstrates the proxy gaming phenomenon.
# Interactive version has a KL budget slider.
#
# Run:
#   python 06_reward_overoptimization.py

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

# ── Synthetic data approximating the overoptimization curve ──
kl_values = np.linspace(0, 25, 200)

# Predicted reward: monotonically increasing (the RM always thinks it's getting better)
predicted_reward = 0.8 * np.sqrt(kl_values) + 0.1 * kl_values

# True human preference: peaks around KL=8, then drops
true_preference = 1.5 * (1 - np.exp(-0.4 * kl_values)) - 0.04 * (kl_values - 8)**2 * (kl_values > 8)
true_preference = np.clip(true_preference, -2, 2)

# Find the peak
peak_idx = np.argmax(true_preference)
peak_kl = kl_values[peak_idx]
peak_val = true_preference[peak_idx]

# ── Matplotlib static ──
fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(kl_values, predicted_reward, color=COLORS['blue'], linewidth=2.5,
        label='RM predicted reward', linestyle='-')
ax.plot(kl_values, true_preference, color=COLORS['green'], linewidth=2.5,
        label='True human preference', linestyle='-')

# Mark the peak and divergence
ax.axvline(x=peak_kl, color=COLORS['orange'], linestyle=':', alpha=0.7)
ax.plot(peak_kl, peak_val, 'o', color=COLORS['orange'], markersize=10, zorder=5)
ax.annotate(f'Peak quality\n(KL ≈ {peak_kl:.0f})', xy=(peak_kl, peak_val),
            xytext=(peak_kl + 4, peak_val + 0.3),
            arrowprops=dict(arrowstyle='->', color=COLORS['orange']),
            fontsize=10, color=COLORS['orange'], fontweight='bold')

# Shade the overoptimization region
ax.fill_between(kl_values[peak_idx:], true_preference[peak_idx:],
                predicted_reward[peak_idx:], alpha=0.15, color=COLORS['red'],
                label='Overoptimization gap')

ax.set_xlabel('KL Divergence from SFT Policy (nats)')
ax.set_ylabel('Quality Score')
ax.set_title('Reward Overoptimization: Predicted vs True Quality')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(0, 25)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR_STATIC / "06_reward_overoptimization.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '06_reward_overoptimization.png'}")

# ── Plotly interactive with KL budget marker ──
fig_plotly = go.Figure()

fig_plotly.add_trace(go.Scatter(
    x=kl_values, y=predicted_reward, mode='lines', name='RM Predicted Reward',
    line=dict(color='#2E86C1', width=3),
    hovertemplate='KL: %{x:.1f}<br>Predicted: %{y:.2f}<extra></extra>'
))
fig_plotly.add_trace(go.Scatter(
    x=kl_values, y=true_preference, mode='lines', name='True Human Preference',
    line=dict(color='#27AE60', width=3),
    hovertemplate='KL: %{x:.1f}<br>True: %{y:.2f}<extra></extra>'
))

# Overoptimization region fill
fig_plotly.add_trace(go.Scatter(
    x=np.concatenate([kl_values[peak_idx:], kl_values[peak_idx:][::-1]]),
    y=np.concatenate([predicted_reward[peak_idx:], true_preference[peak_idx:][::-1]]),
    fill='toself', fillcolor='rgba(231,76,60,0.15)',
    line=dict(color='rgba(0,0,0,0)'),
    name='Overoptimization Gap', showlegend=True,
    hoverinfo='skip'
))

# Peak marker
fig_plotly.add_trace(go.Scatter(
    x=[peak_kl], y=[peak_val], mode='markers+text',
    marker=dict(size=12, color='#E8790C', symbol='circle'),
    text=[f'Peak (KL≈{peak_kl:.0f})'], textposition='top right',
    textfont=dict(color='#E8790C', size=12),
    name='Optimal KL', showlegend=False
))

# KL budget slider
kl_budgets = np.arange(0, 26, 1)
steps = []
for kl_b in kl_budgets:
    step = dict(
        method="relayout",
        args=[{"shapes": [dict(
            type="line", x0=kl_b, x1=kl_b, y0=-2, y1=5,
            line=dict(color="#d2a8ff", width=2, dash="dash"),
        )]}],
        label=f"{kl_b}"
    )
    steps.append(step)

fig_plotly.update_layout(
    template="plotly_dark",
    title=dict(text="Reward Overoptimization: RM Score vs True Quality", font=dict(size=18)),
    xaxis=dict(title="KL Divergence from SFT (nats)", range=[0, 25]),
    yaxis=dict(title="Quality Score", range=[-1.5, 4.5]),
    margin=dict(t=60, b=120, l=50, r=30),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(22,27,34,0.8)"),
    hovermode="x unified",
    sliders=[dict(
        active=8,
        currentvalue=dict(prefix="KL Budget: ", suffix=" nats", font=dict(size=14)),
        pad=dict(t=40),
        steps=steps
    )]
)

save_plotly_html(fig_plotly, OUTPUT_DIR_INTERACTIVE / "06_reward_overoptimization.html")
