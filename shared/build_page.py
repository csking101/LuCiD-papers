#!/usr/bin/env python3
"""Build a docs page for a paper from its page_config.py.

Usage:
    python shared/build_page.py <arxiv_id>

Example:
    python shared/build_page.py 2009.01325

This loads papers/<arxiv_id>/page_config.py, renders shared/template.html
with Jinja2, and writes the result to docs/papers/<arxiv_id>/index.html.
"""

import importlib.util
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def load_config(arxiv_id: str) -> dict:
    """Import PAGE_DATA from papers/<arxiv_id>/page_config.py."""
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "papers" / arxiv_id / "page_config.py"

    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("page_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PAGE_DATA"):
        print(f"Error: PAGE_DATA not found in {config_path}")
        sys.exit(1)

    return module.PAGE_DATA


def build(arxiv_id: str) -> None:
    """Render template with config and write to docs/."""
    repo_root = Path(__file__).resolve().parent.parent
    shared_dir = repo_root / "shared"
    template_path = shared_dir / "template.html"

    if not template_path.exists():
        print(f"Error: {template_path} not found")
        sys.exit(1)

    # Load config
    data = load_config(arxiv_id)

    # Set up Jinja2
    env = Environment(
        loader=FileSystemLoader(str(shared_dir)),
        autoescape=False,  # HTML content is trusted (we write it ourselves)
        keep_trailing_newline=True,
    )
    template = env.get_template("template.html")

    # Render
    html = template.render(**data)

    # Write output
    out_dir = repo_root / "docs" / "papers" / arxiv_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"Built: {out_path}")
    print(f"  Sections: {sum(1 for b in data['content'] if b['type'] == 'note')}")
    print(f"  Vizs:     {sum(1 for b in data['content'] if b['type'] == 'viz')}")
    print(f"  Total lines: {html.count(chr(10)) + 1}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    arxiv_id = sys.argv[1]
    build(arxiv_id)


if __name__ == "__main__":
    main()
