"""
Shared Plotly utilities for LuCiD-papers visualizations.
Uses CDN-based plotly.js to keep HTML files small (~50KB vs ~4.7MB).

Usage:
    from shared.plotly_utils import save_plotly_html
    save_plotly_html(fig, output_path)
"""

from pathlib import Path

PLOTLY_CDN_VERSION = "2.35.2"

# Injected into every Plotly HTML to eliminate default browser margins
# and make the chart fill the full iframe / viewport.
_RESET_CSS = (
    '<style>html,body{margin:0;padding:0;overflow:hidden;'
    'background:#0d1117;width:100%;height:100%}'
    '.plotly-graph-div{width:100%!important;height:100%!important}</style>'
)


def save_plotly_html(fig, output_path):
    """Save a Plotly figure as HTML using CDN-hosted plotly.js.

    This produces ~50KB files instead of ~4.7MB when plotly.js is embedded.
    Requires internet access to view the resulting HTML.

    The output HTML has zero body margin and a dark background so it
    renders cleanly inside iframes without whitespace borders.

    Args:
        fig: A plotly.graph_objects.Figure instance.
        output_path: Path (str or Path) where the HTML file will be written.
    """
    fig.write_html(
        str(output_path),
        include_plotlyjs='cdn',
        default_width='100%',
        default_height='100%',
    )
    # Post-process: inject reset CSS right after <head>
    path = Path(output_path)
    html = path.read_text()
    html = html.replace('<head>', f'<head>{_RESET_CSS}', 1)
    path.write_text(html)
    print(f"Saved: {output_path}")
