# 10 — PPO for LLM Alignment
#
# Tool: Matplotlib + Plotly
# Output: PNG + HTML
#
# Shows PPO in the RLHF context: token-level policy, reward model scoring
# a generation, KL penalty against SFT model, token-level advantages.
# Visualizes how clipping prevents reward hacking.
#
# Run:
#   python 10_ppo_llm_alignment.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def plot_static():
    """Multi-panel figure showing PPO applied to LLM alignment."""
    fig = plt.figure(figsize=(16, 10))

    # ── Panel 1: Token-level policy (top left) ──
    ax1 = fig.add_subplot(2, 2, 1)
    tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']
    # Simulated log-probabilities under PPO policy vs SFT policy
    ppo_logprobs = [-0.3, -0.5, -0.8, -0.2, -0.3, -1.2, -0.1]
    sft_logprobs = [-0.4, -0.6, -0.7, -0.3, -0.3, -0.9, -0.1]

    x = np.arange(len(tokens))
    width = 0.35
    ax1.bar(x - width/2, ppo_logprobs, width, label='PPO Policy', color=COLORS['green'], alpha=0.8)
    ax1.bar(x + width/2, sft_logprobs, width, label='SFT Policy', color=COLORS['blue'], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens, fontsize=10)
    ax1.set_ylabel('Log Probability')
    ax1.set_title('Token-Level Policy Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.axhline(y=0, color='grey', linewidth=0.5)

    # ── Panel 2: Token-level advantages (top right) ──
    ax2 = fig.add_subplot(2, 2, 2)
    advantages = [0.1, 0.3, -0.2, 0.05, 0.0, -0.8, 0.15]
    colors_adv = [COLORS['green'] if a > 0 else COLORS['red'] for a in advantages]
    ax2.bar(x, advantages, color=colors_adv, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tokens, fontsize=10)
    ax2.set_ylabel('Advantage')
    ax2.set_title('Token-Level Advantages', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='grey', linewidth=0.5)
    # Annotate
    ax2.annotate('"mat" has negative\nadvantage: suppress it',
                 xy=(5, -0.8), xytext=(3.5, -0.6),
                 arrowprops=dict(arrowstyle='->', color=COLORS['orange']),
                 fontsize=9, color=COLORS['orange'])

    # ── Panel 3: KL penalty effect (bottom left) ──
    ax3 = fig.add_subplot(2, 2, 3)
    training_steps = np.arange(0, 100)
    reward_no_kl = 5 * (1 - np.exp(-training_steps / 20)) + 0.3 * training_steps / 100
    reward_with_kl = 4 * (1 - np.exp(-training_steps / 25))
    kl_div = 0.02 * training_steps + 0.001 * training_steps**1.5 / 100

    ax3.plot(training_steps, reward_no_kl, color=COLORS['red'], linewidth=2,
             label='Reward (no KL penalty)', linestyle='--')
    ax3.plot(training_steps, reward_with_kl, color=COLORS['green'], linewidth=2,
             label='Reward (with KL penalty)')
    ax3.fill_between(training_steps, reward_no_kl, reward_with_kl,
                      where=reward_no_kl > reward_with_kl, alpha=0.1, color=COLORS['red'])
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Reward Model Score')
    ax3.set_title('KL Penalty Prevents Reward Hacking', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.annotate('Without KL: reward keeps rising\n(reward hacking)',
                 xy=(80, reward_no_kl[80]), xytext=(50, 6),
                 arrowprops=dict(arrowstyle='->', color=COLORS['red']),
                 fontsize=9, color=COLORS['red'])

    # ── Panel 4: The RLHF pipeline (bottom right) ──
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 6)
    ax4.axis('off')
    ax4.set_title('PPO in the RLHF Pipeline', fontsize=12, fontweight='bold')

    # Draw boxes
    boxes = [
        (1, 4.5, 'Prompt', '#58a6ff'),
        (4, 4.5, 'LLM\n(Policy)', '#3fb950'),
        (7.5, 4.5, 'Response', '#d2a8ff'),
        (7.5, 2.5, 'Reward\nModel', '#f0883e'),
        (4, 2.5, 'PPO\nUpdate', '#E74C3C'),
        (1, 2.5, 'KL\nPenalty', '#8B45A6'),
    ]
    for bx, by, text, color in boxes:
        rect = mpatches.FancyBboxPatch((bx - 0.8, by - 0.5), 1.6, 1.0,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.2,
                                        edgecolor=color, linewidth=1.5)
        ax4.add_patch(rect)
        ax4.text(bx, by, text, ha='center', va='center', fontsize=9,
                 color=color, fontweight='bold')

    # Draw arrows
    arrow_props = dict(arrowstyle='->', color='grey', lw=1.5)
    arrows = [
        ((1.8, 4.5), (3.2, 4.5)),   # Prompt -> LLM
        ((4.8, 4.5), (6.7, 4.5)),   # LLM -> Response
        ((7.5, 4.0), (7.5, 3.0)),   # Response -> Reward Model
        ((6.7, 2.5), (4.8, 2.5)),   # Reward -> PPO
        ((3.2, 2.5), (1.8, 2.5)),   # PPO -> KL
        ((4.0, 3.0), (4.0, 4.0)),   # PPO -> LLM (update)
    ]
    for start, end in arrows:
        ax4.annotate('', xy=end, xytext=start, arrowprops=arrow_props)

    plt.tight_layout()
    output_path = OUTPUT_DIR_STATIC / "10_ppo_llm_alignment.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved: {output_path}")


def plot_interactive():
    """Interactive: effect of KL penalty coefficient beta."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Reward Over Training",
                                        "KL Divergence from SFT"])

    betas = np.arange(0.0, 0.55, 0.05)
    training_steps = np.arange(0, 100)

    for i, beta in enumerate(betas):
        # Simulate reward and KL for different beta
        if beta == 0:
            reward = 5 * (1 - np.exp(-training_steps / 20)) + 0.3 * training_steps / 100
            kl = 0.02 * training_steps + 0.001 * training_steps**1.5 / 100
        else:
            damping = 1 / (1 + beta * 10)
            reward = (5 * damping) * (1 - np.exp(-training_steps / (20 + beta * 30)))
            kl = (0.02 / (1 + beta * 5)) * training_steps

        visible = abs(beta - 0.1) < 0.01

        fig.add_trace(go.Scatter(
            x=training_steps.tolist(), y=reward.tolist(), mode='lines',
            line=dict(color='#27AE60', width=2),
            name='Reward', showlegend=False, visible=visible,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=training_steps.tolist(), y=kl.tolist(), mode='lines',
            line=dict(color='#E74C3C', width=2),
            name='KL Divergence', showlegend=False, visible=visible,
        ), row=1, col=2)

    steps = []
    for i, beta in enumerate(betas):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"RLHF Training Dynamics (beta={beta:.2f})"}],
            label=f"{beta:.2f}"
        )
        step["args"][0]["visible"][i * 2] = True
        step["args"][0]["visible"][i * 2 + 1] = True
        steps.append(step)

    fig.update_layout(
        sliders=[dict(
            active=2,  # beta=0.1
            currentvalue={"prefix": "KL penalty beta = "},
            pad={"t": 50},
            steps=steps,
        )],
        title="RLHF Training Dynamics (beta=0.10)",
        template="plotly_dark",
        margin=dict(t=60, b=80, l=50, r=30),
    )

    fig.update_xaxes(title_text="Training Steps", row=1, col=1)
    fig.update_xaxes(title_text="Training Steps", row=1, col=2)
    fig.update_yaxes(title_text="Reward Model Score", row=1, col=1)
    fig.update_yaxes(title_text="KL from SFT Model", row=1, col=2)

    output_path = OUTPUT_DIR_INTERACTIVE / "10_ppo_llm_alignment.html"
    save_plotly_html(fig, output_path)


if __name__ == "__main__":
    plot_static()
    plot_interactive()
