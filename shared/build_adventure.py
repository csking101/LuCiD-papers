#!/usr/bin/env python3
"""Build a docs page for a coding adventure from its page_config.py.

Usage:
    python shared/build_adventure.py <adventure_num>

Example:
    python shared/build_adventure.py 01

This finds the adventure directory matching the number, loads its
page_config.py, renders shared/adventure_template.html with Jinja2,
and writes the result to docs/adventures/<num>/index.html.
"""

import importlib.util
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def find_adventure_dir(num: str) -> Path:
    """Find the coding-adventures directory matching *num* (e.g. '01')."""
    repo_root = Path(__file__).resolve().parent.parent
    adventures_root = repo_root / "coding-adventures"

    for d in sorted(adventures_root.iterdir()):
        if d.is_dir() and d.name.startswith(f"{num}-"):
            return d

    # Exact match fallback
    exact = adventures_root / num
    if exact.is_dir():
        return exact

    print(f"Error: No adventure directory found for number '{num}' in {adventures_root}")
    sys.exit(1)


def load_config(adventure_dir: Path) -> dict:
    """Import PAGE_DATA from <adventure_dir>/page_config.py."""
    config_path = adventure_dir / "page_config.py"

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


def build(num: str) -> None:
    """Render adventure template with config and write to docs/."""
    repo_root = Path(__file__).resolve().parent.parent
    shared_dir = repo_root / "shared"
    template_path = shared_dir / "adventure_template.html"

    if not template_path.exists():
        print(f"Error: {template_path} not found")
        sys.exit(1)

    # Find adventure dir and load config
    adventure_dir = find_adventure_dir(num)
    data = load_config(adventure_dir)

    # Set up Jinja2
    env = Environment(
        loader=FileSystemLoader(str(shared_dir)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = env.get_template("adventure_template.html")

    # Render
    html = template.render(**data)

    # Write output
    out_dir = repo_root / "docs" / "adventures" / num
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")

    n_notes = sum(1 for b in data["content"] if b["type"] == "note")
    n_screenshots = sum(1 for b in data["content"] if b["type"] == "screenshot")

    print(f"Built: {out_path}")
    print(f"  Sections: {n_notes}")
    print(f"  Screenshots: {n_screenshots}")
    print(f"  Total lines: {html.count(chr(10)) + 1}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    num = sys.argv[1]
    build(num)


if __name__ == "__main__":
    main()
