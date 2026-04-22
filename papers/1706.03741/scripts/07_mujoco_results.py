"""
Visualization 7: MuJoCo Results
Reproduces Figure 2 from "Deep RL from Human Preferences" (Christiano et al., 2017)

Compares:
- RL with true reward (orange)
- RLHF with real human feedback (purple)
- RLHF with synthetic oracle feedback at 350, 700, 1400 labels (blues)

Data is approximated from the paper's Figure 2 curves.
"""

import sys
from pathlib import Path

# Add repo root for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt

from shared.style import apply_style, COLORS as STYLE_COLORS

apply_style()

# Color scheme matching the paper
COLORS = {
    'true_reward': '#E8790C',      # orange
    'human_feedback': '#8B45A6',   # purple
    'synthetic_1400': '#1A5276',   # dark blue
    'synthetic_700': '#2E86C1',    # medium blue
    'synthetic_350': '#85C1E9',    # light blue
}

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "static"

# --- Synthetic Data (approximated from paper's Figure 2) ---


def smooth(y, window=3):
    """Simple moving average for smoother curves."""
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')


def generate_curve(x, final_val, noise_std=0.05, warmup=0.2, curve_type='sigmoid'):
    """Generate a plausible training curve."""
    rng = np.random.RandomState(hash(str(final_val) + curve_type) % 2**31)
    if curve_type == 'sigmoid':
        raw = final_val / (1 + np.exp(-8 * (x - 0.35)))
    elif curve_type == 'log':
        raw = final_val * np.log(1 + 5 * x) / np.log(6)
    elif curve_type == 'linear':
        raw = final_val * x
    elif curve_type == 'slow':
        raw = final_val * (1 - np.exp(-3 * x))
    else:
        raw = final_val * x

    noise = rng.normal(0, noise_std * abs(final_val) + 0.01, len(x))
    raw = raw + noise
    # warmup phase
    warmup_mask = x < warmup
    raw[warmup_mask] *= x[warmup_mask] / warmup
    return smooth(raw, window=5)


# Task definitions with approximate final performance values
tasks = {
    'Reacher': {
        'x_label': 'Time Steps (millions)',
        'x_max': 2,
        'y_label': 'Reward',
        'true_reward': (-5, 'sigmoid'),
        'human': (-7, 'sigmoid'),
        'syn_1400': (-4, 'sigmoid'),
        'syn_700': (-5, 'sigmoid'),
        'syn_350': (-8, 'sigmoid'),
        'y_range': (-50, 0),
        'reward_offset': -30,
    },
    'Half Cheetah': {
        'x_label': 'Time Steps (millions)',
        'x_max': 5,
        'y_label': 'Reward',
        'true_reward': (3000, 'sigmoid'),
        'human': (2700, 'sigmoid'),
        'syn_1400': (3200, 'sigmoid'),
        'syn_700': (2800, 'sigmoid'),
        'syn_350': (2000, 'sigmoid'),
        'y_range': (-500, 4000),
        'reward_offset': 0,
    },
    'Hopper': {
        'x_label': 'Time Steps (millions)',
        'x_max': 4,
        'y_label': 'Reward',
        'true_reward': (2500, 'sigmoid'),
        'human': (2200, 'log'),
        'syn_1400': (2600, 'sigmoid'),
        'syn_700': (2300, 'sigmoid'),
        'syn_350': (1800, 'slow'),
        'y_range': (0, 3500),
        'reward_offset': 0,
    },
    'Walker': {
        'x_label': 'Time Steps (millions)',
        'x_max': 10,
        'y_label': 'Reward',
        'true_reward': (3500, 'sigmoid'),
        'human': (3000, 'sigmoid'),
        'syn_1400': (3800, 'sigmoid'),
        'syn_700': (3200, 'sigmoid'),
        'syn_350': (2500, 'slow'),
        'y_range': (0, 5000),
        'reward_offset': 0,
    },
    'Swimmer': {
        'x_label': 'Time Steps (millions)',
        'x_max': 5,
        'y_label': 'Reward',
        'true_reward': (250, 'log'),
        'human': (200, 'log'),
        'syn_1400': (280, 'log'),
        'syn_700': (230, 'log'),
        'syn_350': (170, 'slow'),
        'y_range': (0, 350),
        'reward_offset': 0,
    },
    'Ant': {
        'x_label': 'Time Steps (millions)',
        'x_max': 10,
        'y_label': 'Reward',
        'true_reward': (3500, 'sigmoid'),
        'human': (3800, 'sigmoid'),
        'syn_1400': (3600, 'sigmoid'),
        'syn_700': (3200, 'sigmoid'),
        'syn_350': (2200, 'slow'),
        'y_range': (0, 5000),
        'reward_offset': 0,
    },
    'Pendulum': {
        'x_label': 'Time Steps (millions)',
        'x_max': 2,
        'y_label': 'Reward',
        'true_reward': (-200, 'sigmoid'),
        'human': (-250, 'sigmoid'),
        'syn_1400': (-180, 'sigmoid'),
        'syn_700': (-220, 'sigmoid'),
        'syn_350': (-350, 'slow'),
        'y_range': (-1200, 0),
        'reward_offset': -800,
    },
    'Double Pendulum': {
        'x_label': 'Time Steps (millions)',
        'x_max': 5,
        'y_label': 'Reward',
        'true_reward': (-200, 'sigmoid'),
        'human': (-300, 'sigmoid'),
        'syn_1400': (-180, 'sigmoid'),
        'syn_700': (-250, 'sigmoid'),
        'syn_350': (-400, 'slow'),
        'y_range': (-1500, 0),
        'reward_offset': -900,
    },
}


def plot_mujoco_results():
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        'Deep RL from Human Preferences — MuJoCo Results (Figure 2)',
        fontsize=16, fontweight='bold', y=0.98
    )
    fig.text(
        0.5, 0.93,
        'Comparison of true reward RL, human feedback RLHF, and synthetic oracle RLHF across 8 robotics tasks',
        ha='center', fontsize=11, color='#555555'
    )

    task_names = list(tasks.keys())

    for idx, task_name in enumerate(task_names):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]
        task = tasks[task_name]

        n_points = 100
        x = np.linspace(0, 1, n_points)
        x_display = x * task['x_max']

        # Generate and plot each curve
        methods = [
            ('true_reward', 'RL (true reward)', COLORS['true_reward'], 2.5, '-'),
            ('human', 'Human feedback (700)', COLORS['human_feedback'], 2.0, '-'),
            ('syn_1400', 'Synthetic (1400)', COLORS['synthetic_1400'], 1.5, '--'),
            ('syn_700', 'Synthetic (700)', COLORS['synthetic_700'], 1.5, '--'),
            ('syn_350', 'Synthetic (350)', COLORS['synthetic_350'], 1.5, '--'),
        ]

        for key, label, color, lw, ls in methods:
            final_val, curve_type = task[key]
            offset = task.get('reward_offset', 0)
            y = generate_curve(x, final_val - offset, noise_std=0.03, curve_type=curve_type) + offset
            ax.plot(x_display, y, color=color, linewidth=lw, linestyle=ls, label=label, alpha=0.9)

        ax.set_title(task_name, pad=8)
        ax.set_xlabel(task['x_label'], fontsize=8)
        ax.set_ylabel(task['y_label'], fontsize=8)
        ax.set_ylim(task['y_range'])
        ax.tick_params(labelsize=7)

    # Single legend for the whole figure
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
    output_path = OUTPUT_DIR / "07_mujoco_results.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    plot_mujoco_results()
