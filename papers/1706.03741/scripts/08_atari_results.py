"""
Visualization 8: Atari Results
Reproduces Figure 3 from "Deep RL from Human Preferences" (Christiano et al., 2017)

Compares across 7 Atari games:
- RL with true reward (orange)
- RLHF with real human feedback (purple)
- RLHF with synthetic oracle feedback at 350, 700, 1400 labels (blues)

Data is approximated from the paper's Figure 3 curves.
"""

import sys
from pathlib import Path

# Add repo root for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt

from shared.style import apply_style

apply_style()

COLORS = {
    'true_reward': '#E8790C',
    'human_feedback': '#8B45A6',
    'synthetic_5500': '#1A5276',
    'synthetic_3300': '#2E86C1',
    'synthetic_700': '#85C1E9',
}

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "static"


def smooth(y, window=5):
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')


def generate_curve(x, final_val, noise_std=0.04, curve_type='sigmoid', seed_extra=''):
    rng = np.random.RandomState(hash(str(final_val) + curve_type + seed_extra) % 2**31)
    if curve_type == 'sigmoid':
        raw = final_val / (1 + np.exp(-8 * (x - 0.35)))
    elif curve_type == 'log':
        raw = final_val * np.log(1 + 5 * x) / np.log(6)
    elif curve_type == 'slow':
        raw = final_val * (1 - np.exp(-2.5 * x))
    elif curve_type == 'fast':
        raw = final_val * (1 - np.exp(-5 * x))
    elif curve_type == 'plateau':
        raw = final_val * np.minimum(1.0, 2.5 * x)
    elif curve_type == 'noisy_climb':
        raw = final_val * x ** 0.7
    else:
        raw = final_val * x

    noise = rng.normal(0, noise_std * abs(final_val) + 0.5, len(x))
    raw = raw + noise
    warmup_mask = x < 0.15
    raw[warmup_mask] *= x[warmup_mask] / 0.15
    return smooth(raw, window=5)


# Atari games with approximate final performance from Figure 3
tasks = {
    'Beamrider': {
        'x_frames': 50,
        'true_reward': (4500, 'sigmoid'),
        'human': (3000, 'sigmoid'),
        'syn_5500': (4200, 'sigmoid'),
        'syn_3300': (3800, 'slow'),
        'syn_700': (2000, 'slow'),
        'y_range': (-200, 6000),
    },
    'Breakout': {
        'x_frames': 50,
        'true_reward': (300, 'sigmoid'),
        'human': (50, 'slow'),
        'syn_5500': (120, 'slow'),
        'syn_3300': (80, 'slow'),
        'syn_700': (30, 'noisy_climb'),
        'y_range': (-10, 400),
    },
    'Enduro': {
        'x_frames': 50,
        'true_reward': (400, 'slow'),
        'human': (450, 'sigmoid'),
        'syn_5500': (350, 'slow'),
        'syn_3300': (250, 'slow'),
        'syn_700': (100, 'noisy_climb'),
        'y_range': (-20, 600),
    },
    'Pong': {
        'x_frames': 50,
        'true_reward': (20, 'sigmoid'),
        'human': (15, 'sigmoid'),
        'syn_5500': (20, 'sigmoid'),
        'syn_3300': (18, 'sigmoid'),
        'syn_700': (5, 'slow'),
        'y_range': (-21, 21),
    },
    'Qbert': {
        'x_frames': 50,
        'true_reward': (14000, 'sigmoid'),
        'human': (2000, 'slow'),
        'syn_5500': (12000, 'slow'),
        'syn_3300': (8000, 'slow'),
        'syn_700': (3000, 'noisy_climb'),
        'y_range': (-500, 18000),
    },
    'Seaquest': {
        'x_frames': 50,
        'true_reward': (1200, 'sigmoid'),
        'human': (800, 'slow'),
        'syn_5500': (1100, 'slow'),
        'syn_3300': (900, 'slow'),
        'syn_700': (500, 'noisy_climb'),
        'y_range': (-50, 1800),
    },
    'Space Invaders': {
        'x_frames': 50,
        'true_reward': (700, 'sigmoid'),
        'human': (400, 'slow'),
        'syn_5500': (500, 'slow'),
        'syn_3300': (400, 'slow'),
        'syn_700': (250, 'noisy_climb'),
        'y_range': (-30, 1000),
    },
}


def plot_atari_results():
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        'Deep RL from Human Preferences — Atari Results (Figure 3)',
        fontsize=16, fontweight='bold', y=0.98
    )
    fig.text(
        0.5, 0.93,
        'Comparison of true reward RL, human feedback RLHF, and synthetic oracle RLHF across 7 Atari games',
        ha='center', fontsize=11, color='#555555'
    )

    task_names = list(tasks.keys())

    for idx, task_name in enumerate(task_names):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]
        task = tasks[task_name]

        n_points = 120
        x = np.linspace(0, 1, n_points)
        x_display = x * task['x_frames']

        methods = [
            ('true_reward', 'RL (true reward)', COLORS['true_reward'], 2.5, '-'),
            ('human', 'Human feedback (5500)', COLORS['human_feedback'], 2.0, '-'),
            ('syn_5500', 'Synthetic (5500)', COLORS['synthetic_5500'], 1.5, '--'),
            ('syn_3300', 'Synthetic (3300)', COLORS['synthetic_3300'], 1.5, '--'),
            ('syn_700', 'Synthetic (700)', COLORS['synthetic_700'], 1.5, '--'),
        ]

        for key, label, color, lw, ls in methods:
            final_val, curve_type = task[key]
            y = generate_curve(x, final_val, noise_std=0.04, curve_type=curve_type, seed_extra=task_name)
            ax.plot(x_display, y, color=color, linewidth=lw, linestyle=ls, label=label, alpha=0.9)

        ax.set_title(task_name, pad=8)
        ax.set_xlabel('Frames (millions)', fontsize=8)
        ax.set_ylabel('Score', fontsize=8)
        ax.set_ylim(task['y_range'])
        ax.tick_params(labelsize=7)

    # Hide the 8th subplot (only 7 games)
    axes[1, 3].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=5,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.91])
    output_path = OUTPUT_DIR / "08_atari_results.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    plot_atari_results()
