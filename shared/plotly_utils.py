"""
Shared Plotly utilities for LuCiD-papers visualizations.
Uses CDN-based plotly.js to keep HTML files small (~50KB vs ~4.7MB).

Usage:
    from shared.plotly_utils import save_plotly_html
    save_plotly_html(fig, output_path)
"""

PLOTLY_CDN_VERSION = "2.35.2"


def save_plotly_html(fig, output_path):
    """Save a Plotly figure as HTML using CDN-hosted plotly.js.

    This produces ~50KB files instead of ~4.7MB when plotly.js is embedded.
    Requires internet access to view the resulting HTML.

    Args:
        fig: A plotly.graph_objects.Figure instance.
        output_path: Path (str or Path) where the HTML file will be written.
    """
    fig.write_html(
        str(output_path),
        include_plotlyjs='cdn',
    )
    print(f"Saved: {output_path}")
