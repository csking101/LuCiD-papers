# 07 — KL Coefficient Effect
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Shows how varying β trades off reward model score vs KL divergence from SFT.
# Visualizes the Pareto frontier of the tradeoff.
#
# Run:
#   python 07_kl_coefficient_effect.py

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

# ── Synthetic data: β vs resulting KL and reward ──
betas = np.array([0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
# Higher β → lower KL (policy stays closer to SFT)
kl_divergence = 25 * np.exp(-2.5 * betas) + 0.5
# Higher β → lower reward (less aggressive optimization)
rm_reward = 2.8 * (1 - np.exp(-8 * kl_divergence / 25))
# True human preference: peaks at moderate KL
true_pref = 1.8 * (1 - np.exp(-0.5 * kl_divergence)) - 0.02 * np.maximum(kl_divergence - 8, 0)**2
true_pref = np.clip(true_pref, -0.5, 2)

# ── Matplotlib static ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: β vs KL
ax1.semilogx(betas, kl_divergence, 'o-', color=COLORS['blue'], linewidth=2, markersize=7)
ax1.set_xlabel('β (KL coefficient)')
ax1.set_ylabel('KL Divergence (nats)')
ax1.set_title('β Controls Policy Drift')
ax1.grid(True, alpha=0.3)
ax1.axhspan(6, 10, alpha=0.1, color=COLORS['green'], label='Sweet spot')
ax1.legend(fontsize=9)

# Right: KL vs Quality (Pareto frontier)
ax2.plot(kl_divergence, rm_reward, 's-', color=COLORS['blue'], linewidth=2,
         markersize=7, label='RM reward')
ax2.plot(kl_divergence, true_pref, 'o-', color=COLORS['green'], linewidth=2,
         markersize=7, label='True preference')
ax2.set_xlabel('KL Divergence (nats)')
ax2.set_ylabel('Quality Score')
ax2.set_title('KL vs Quality: The Pareto Frontier')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Annotate optimal β
opt_idx = np.argmax(true_pref)
ax2.plot(kl_divergence[opt_idx], true_pref[opt_idx], '*', color=COLORS['orange'],
         markersize=15, zorder=5)
ax2.annotate(f'β={betas[opt_idx]}', xy=(kl_divergence[opt_idx], true_pref[opt_idx]),
             xytext=(kl_divergence[opt_idx] + 3, true_pref[opt_idx] + 0.3),
             arrowprops=dict(arrowstyle='->', color=COLORS['orange']),
             fontsize=10, color=COLORS['orange'], fontweight='bold')

fig.tight_layout()
fig.savefig(OUTPUT_DIR_STATIC / "07_kl_coefficient_effect.png", dpi=200,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {OUTPUT_DIR_STATIC / '07_kl_coefficient_effect.png'}")

# ── Plotly interactive ──
fig_plotly = go.Figure()

fig_plotly.add_trace(go.Scatter(
    x=kl_divergence, y=rm_reward, mode='lines+markers+text',
    name='RM Reward',
    line=dict(color='#2E86C1', width=2.5),
    marker=dict(size=9, symbol='square'),
    text=[f'β={b}' for b in betas],
    textposition='top center', textfont=dict(size=9, color='#8b949e'),
    hovertemplate='KL: %{x:.1f} nats<br>RM Reward: %{y:.2f}<br>β=%{text}<extra></extra>'
))
fig_plotly.add_trace(go.Scatter(
    x=kl_divergence, y=true_pref, mode='lines+markers',
    name='True Human Preference',
    line=dict(color='#27AE60', width=2.5),
    marker=dict(size=9),
    hovertemplate='KL: %{x:.1f} nats<br>True Pref: %{y:.2f}<extra></extra>'
))

# Optimal point
fig_plotly.add_trace(go.Scatter(
    x=[kl_divergence[opt_idx]], y=[true_pref[opt_idx]],
    mode='markers', name=f'Optimal (β={betas[opt_idx]})',
    marker=dict(size=16, color='#E8790C', symbol='star'),
))

fig_plotly.update_layout(
    template="plotly_dark",
    title=dict(text="KL Coefficient β: Trading Off Reward vs Stability", font=dict(size=18)),
    xaxis=dict(title="KL Divergence from SFT (nats)"),
    yaxis=dict(title="Quality Score"),
    margin=dict(t=60, b=80, l=50, r=30),
    legend=dict(x=0.6, y=0.98, bgcolor="rgba(22,27,34,0.8)"),
    hovermode="x unified",
)

save_plotly_html(fig_plotly, OUTPUT_DIR_INTERACTIVE / "07_kl_coefficient_effect.html")
