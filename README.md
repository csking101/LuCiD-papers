# LuCiD-papers

> Making AI papers **lu**cid -- **L**earn, **C**ode, **D**ocument

A collection of visualizations, animations, and interactive demos for AI/ML research papers, following the LCD framework.

**Live demos:** [csking101.github.io/LuCiD-papers](https://csking101.github.io/LuCiD-papers/)

## Papers

| # | Paper | Title | Track | Status | Static | Interactive | Animated |
|---|-------|-------|-------|--------|--------|-------------|----------|
| 1 | [1706.03741](papers/1706.03741/) | Deep RL from Human Preferences | Alignment | Done | 5 PNGs | 4 HTMLs | 4 MP4s |
| 2 | [1707.06347](https://arxiv.org/abs/1707.06347) | Proximal Policy Optimization (PPO) | Alignment | Learning | -- | -- | -- |
| 3 | [2009.01325](https://arxiv.org/abs/2009.01325) | Learning to Summarize from Human Feedback | Alignment | Upcoming | -- | -- | -- |
| 4 | [2203.02155](https://arxiv.org/abs/2203.02155) | InstructGPT | Alignment | Upcoming | -- | -- | -- |
| 5 | [2305.18290](https://arxiv.org/abs/2305.18290) | Direct Preference Optimization (DPO) | Alignment | Upcoming | -- | -- | -- |
| 6 | [2402.03300](https://arxiv.org/abs/2402.03300) | Self-Play Fine-Tuning (SPIN) | Alignment | Upcoming | -- | -- | -- |
| 7 | [2405.17247](https://arxiv.org/abs/2405.17247) | An Introduction to Vision-Language Modeling | VLM | Learning | -- | -- | -- |

## Roadmap

Two parallel reading tracks, each studied through the LCD framework:

**RL/Alignment Track** -- How to align language models with human preferences:

1706.03741 (RLHF) → 1707.06347 (PPO) → 2009.01325 (Summarize) → 2203.02155 (InstructGPT) → 2305.18290 (DPO) → 2402.03300 (SPIN)

**VLM Track** -- Vision-language models from contrastive to generative:

2405.17247 (Survey) → core papers TBD

## LCD Framework

Each paper is studied through three phases:

- **Learn** -- Read, understand, and take notes on the paper
- **Code** -- Write visualization code hands-on -- practice by building static figures, interactive demos, and animations
- **Document** -- Publish and share the work -- GitHub Pages demos, structured notes, and paper walkthroughs

Paper notes live in a separate Obsidian vault. This repo holds the visualization code and outputs.

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
