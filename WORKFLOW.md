# LCD Workflow Reference

> Repeatable process for each new paper. AI reads this at session start.
> Master reading list: `~/Desktop/everything/Information/AI/Reading List.md`

## Directory Template

Create these for each new paper (replace `ARXIV_ID`):

```
papers/ARXIV_ID/
├── README.md                    # Viz index table + CLI commands to regenerate
├── scripts/
│   ├── 01_name.py               # Numbered 01-10, one per visualization
│   └── ...
├── implementation/              # Optional — paper-specific runnable code
│   ├── __init__.py
│   ├── config.py
│   └── ...
└── output/
    ├── static/                  # PNGs from matplotlib scripts
    ├── interactive/             # Plotly HTMLs from dual-output scripts
    └── animations/              # Manim MP4s + build artifacts (gitignored)

docs/papers/ARXIV_ID/
├── index.html                   # Full paper notes page (single-page app)
├── *.mp4                        # Manim animations (regular git, NOT LFS)
├── *.html                       # Interactive Plotly demos (copied from output/interactive/)
└── *.json                       # Implementation demo data (if applicable)
```

## Phase 1: Learn

- [ ] Read the paper on arXiv
- [ ] Create Obsidian note at `~/Desktop/everything/Information/AI/Papers/.../ARXIV_ID - Title.md`
- [ ] Structure notes by paper sections (Abstract, Introduction, Method, Results, Ablations, Conclusion)
- [ ] Add follow-up reading items to the "Things to Read" section
- [ ] Add AI Section with session log (see template below)
- [ ] Update Reading List status: `Pending` → `Learn (Reading)`

## Phase 2: Code (Visualizations)

- [ ] Create `papers/ARXIV_ID/scripts/` directory
- [ ] Write 10 visualization scripts (01-10), selecting tools per this guide:

| Type | Tool | Output | Use when |
|------|------|--------|----------|
| Animated concept walkthrough | Manim | MP4 | Explaining a multi-step process |
| Static figure (reproducing paper) | Matplotlib | PNG | Bar charts, learning curves, multi-panel |
| Interactive exploration | Matplotlib + Plotly | PNG + HTML | Sliders, 3D surfaces, hover details |
| Self-contained simulation | Plotly/JS | HTML | User interacts with the algorithm |

- [ ] All scripts import shared utilities:
  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
  from shared.style import apply_style, COLORS
  from shared.plotly_utils import save_plotly_html
  ```
- [ ] Matplotlib scripts: call `apply_style()` before plotting, save PNGs to `../output/static/`
- [ ] Plotly scripts: use `save_plotly_html(fig, path)` to save HTMLs to `../output/interactive/`
- [ ] Manim scripts: run with `manim -ql --media_dir ../output/animations script.py ClassName`
- [ ] Manim scripts: add `atexit` hook to auto-copy final MP4 to `docs/papers/ARXIV_ID/`
- [ ] Copy interactive HTMLs to `docs/papers/ARXIV_ID/`

## Phase 3: Implement (Optional)

Only if the paper warrants a runnable implementation (e.g., RLHF cat grid-world).

- [ ] Create `papers/ARXIV_ID/implementation/` with modular Python files
- [ ] Add `config.py` with all hyperparameters
- [ ] Add `train.py` as the main entry point
- [ ] Add `export.py` to serialize results to JSON for the browser demo
- [ ] Run training, then `export.py` to produce `results.json`
- [ ] Copy JSON to `docs/papers/ARXIV_ID/`

## Phase 4: Document (GitHub Pages)

- [ ] Create `docs/papers/ARXIV_ID/index.html` — single-page app with:
  - Dark theme (`#0d1117` background, GitHub-dark palette)
  - Sticky sidebar ToC (hidden < 1400px) + nav bar (shown < 1400px)
  - Scroll progress bar (gradient: `#58a6ff` → `#d2a8ff` → `#f0883e`)
  - MathJax 3 CDN for LaTeX
  - Plotly CDN (only if inline charts needed)
  - Sections interleaving notes and visualizations
  - TL;DR box at top
  - Callout boxes: `.callout-key` (orange), `.callout-insight` (purple), `.callout-note` (blue)
  - Formal definition blocks: `.formal-def` (purple left border)
  - Viz sections: `.viz-section` with tool-specific left border colors:
    - Manim: `#3fb950` (green)
    - Matplotlib: `#58a6ff` (blue)
    - Plotly: `#f0883e` (orange)
    - Multi-tool: `#d2a8ff` (purple)
  - Fade-in animations (IntersectionObserver)
  - Fullscreen modal for iframes (Esc to close)
  - Lazy video loading
  - Back-to-top button
  - Responsive down to 480px
  - Footer with attribution + links
- [ ] Embed visualizations:
  - MP4s: `<video>` tags, files served from same directory
  - PNGs: `<img>` from `raw.githubusercontent.com/csking101/LuCiD-papers/main/papers/ARXIV_ID/output/static/`
  - Interactive HTMLs: `<iframe>` from same directory, with fullscreen button
  - Implementation demo: inline Canvas + Plotly, loading from JSON via `fetch()`
- [ ] If implementation demo exists, add it as a section before the last visualization
- [ ] Write `papers/ARXIV_ID/README.md` with viz index table and CLI regeneration commands
- [ ] Add paper card to `docs/index.html` landing page (status badge: Done)
- [ ] Update landing page status badge from "Learning" / "Coming Soon" to "Done"

## Phase 5: Deploy

- [ ] Verify `.gitattributes` LFS exemption covers `docs/**/*.mp4`
- [ ] Verify `.gitignore` excludes `partial_movie_files/`, `texts/`, `__pycache__/`
- [ ] Commit with descriptive message
- [ ] Push to `main` — GitHub Pages auto-deploys from `docs/`
- [ ] Verify site at `https://csking101.github.io/LuCiD-papers/papers/ARXIV_ID/`

## Phase 6: Cross-link

- [ ] Update Obsidian note AI Section: status → `Done`, visualization count, repo link, session log
- [ ] Update Reading List status: `Learn (Reading)` → `Done`
- [ ] Add any new "Things to Read" references from the paper to appropriate Reading List tiers

---

## Conventions

**Manim**: Use `Text()` not `MathTex()` — no LaTeX installed.

**PyTorch**: Force `DEVICE = "cpu"` for small-scale implementations. CUDA kernel-launch overhead on tiny tensors causes catastrophic slowdowns (~240x on the grid-world).

**Plotly HTML export**: Always use `save_plotly_html()` from `shared/plotly_utils.py`. This uses CDN (~50KB) instead of embedding plotly.js (~4.7MB).

**Static PNGs on GitHub Pages**: Load from `raw.githubusercontent.com` URLs, not stored in `docs/`. This keeps the `docs/` directory small. Only MP4s, HTMLs, and JSONs go in `docs/`.

**MP4 LFS rules**: `*.mp4` tracked by LFS globally, but `docs/**/*.mp4` exempted (GitHub Pages can't serve LFS pointers). Manim build artifacts in `papers/*/output/animations/` are gitignored — only the copies in `docs/` are committed.

**Footer attribution**: "Notes by Chinmaya Sahu. Content polish, code, visualizations, and web design by AI."

**Obsidian vault location**: `~/Desktop/everything/` — paper notes live here, separate from the code repo.

**Reading List**: `~/Desktop/everything/Information/AI/Reading List.md` — single source of truth for reading items and status.

---

## Templates

### Obsidian AI Section

```markdown
## AI Section

> Maintained by AI to track context and links across the LCD workflow.

**Status**: [Pending | Learn (Reading) | Code | Document | Done]
**Repo**: papers/ARXIV_ID/ in LuCiD-papers
**Reading List**: [[Reading List]] — item #N in Tier T
**Visualizations**: 0/10 — plan TBD pending paper discussion
**Related Papers**: [[ARXIV_ID - Title]] (relationship)

### Session Log
- **Session 1**: ...
```

### Script Header

```python
# NN — Descriptive Title
#
# Tool: Manim | Matplotlib | Matplotlib+Plotly | Plotly
# Output: MP4 | PNG | PNG+HTML | HTML
#
# What this visualizes and why it matters for understanding the paper.
#
# Run:
#   python NN_name.py                          # for static/interactive
#   manim -ql --media_dir ../output/animations NN_name.py ClassName  # for Manim
```

### Paper README.md

```markdown
# ARXIV_ID — Paper Title

[Paper](https://arxiv.org/abs/ARXIV_ID) |
[Notes](https://csking101.github.io/LuCiD-papers/papers/ARXIV_ID/) |
[Implementation](implementation/)

## Visualizations

| # | Name | Tool | Output | Description |
|---|------|------|--------|-------------|
| 01 | ... | Manim | MP4 | ... |
| 02 | ... | Matplotlib | PNG | ... |

## Regenerate

\`\`\`bash
cd scripts
python 04_name.py
manim -ql --media_dir ../output/animations 01_name.py ClassName
\`\`\`
```
