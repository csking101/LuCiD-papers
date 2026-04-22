"""
Shared matplotlib styling for LuCiD-papers visualizations.
Provides a consistent visual theme across all papers and charts.

Usage:
    from shared.style import apply_style, COLORS
    apply_style()
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


# Common color palette
COLORS = {
    'blue': '#2E86C1',
    'green': '#27AE60',
    'red': '#E74C3C',
    'orange': '#E8790C',
    'purple': '#8B45A6',
    'teal': '#1ABC9C',
    'yellow': '#F39C12',
    'grey': '#95A5A6',
    'dark_blue': '#1A5276',
    'light_blue': '#85C1E9',
}


def apply_style():
    """Apply the standard LuCiD-papers matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'figure.facecolor': '#fafafa',
        'axes.facecolor': '#fafafa',
        'axes.edgecolor': '#cccccc',
        'grid.alpha': 0.3,
    })
