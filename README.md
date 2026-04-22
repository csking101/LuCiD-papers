# LuCiD-papers

> Making AI papers **lu**cid -- **L**earn, **C**ode, **D**ocument

A collection of visualizations, animations, and interactive demos for AI/ML research papers, following the LCD framework.

## Papers

| # | Paper | Title | Static | Interactive | Animated |
|---|-------|-------|--------|-------------|----------|
| 1 | [1706.03741](papers/1706.03741/) | Deep RL from Human Preferences | 5 PNGs | 4 HTMLs | 4 MP4s |

## LCD Framework

Each paper is studied through three phases:

- **Learn** -- Read, understand, and take notes on the paper
- **Code** -- Build visualizations that explain the key concepts
- **Document** -- Write comprehensive notes linking insights back to the source material

Paper notes live in a separate [Obsidian vault](https://github.com/csking101/everything). This repo holds the visualization code and outputs.

## Repository Structure

```
LuCiD-papers/
├── shared/                    # Reusable utilities across all papers
│   ├── style.py               # Common matplotlib theme
│   └── plotly_utils.py        # CDN-based Plotly HTML export
├── papers/
│   └── {arxiv_id}/
│       ├── README.md           # Paper metadata + visualization index
│       ├── scripts/            # Python scripts (matplotlib, plotly, manim)
│       └── output/
│           ├── static/         # PNG figures
│           ├── interactive/    # HTML (Plotly CDN-based, ~50KB each)
│           └── animations/     # Manim MP4 videos
├── docs/
│   └── index.html             # GitHub Pages landing page
└── requirements.txt
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Matplotlib** | Static publication-quality figures |
| **Plotly** | Interactive browser-based explorations (CDN-based for small file sizes) |
| **Manim** | Animated mathematical concept walkthroughs |

## Getting Started

```bash
# Clone
git clone https://github.com/csking101/LuCiD-papers.git
cd LuCiD-papers

# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate all outputs for a paper
cd papers/1706.03741/scripts
python 04_bradley_terry.py                # static + interactive
manim -ql --media_dir ../output/animations 01_system_architecture.py SystemArchitecture  # animation
```

## License

MIT
