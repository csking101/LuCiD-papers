# Design & Implementation Guide

> Technical reference for building LuCiD-papers visualizations and docs pages.
> Covers every hard-won convention so future pages ship clean on the first pass.
> Companion to `WORKFLOW.md` (which covers the *process*; this covers *how to build*).

---

## 1. Manim Animations

### LaTeX: always use `MathTex`

LaTeX is installed (`texlive-latex-base`, `texlive-latex-extra`, `texlive-fonts-recommended`,
`cm-super`, `dvipng`). **All mathematical expressions must use `MathTex()`**, never `Text()`.

```python
# GOOD
eq = MathTex(r"\nabla_\theta J(\theta) = \mathbb{E}\!\left[\hat{A}_t\right]",
             font_size=36, color=WHITE)

# BAD — renders as monospace ASCII, no real math typesetting
eq = Text("grad J(theta) = E[A_t]", font_size=22, color=WHITE)
```

### Font for descriptive text

Use `"Latin Modern Roman"` for all `Text()` elements to match the LaTeX aesthetic:

```python
_FONT = "Latin Modern Roman"

title = Text("Policy Gradient: The Intuition", font_size=36, weight=BOLD,
             color=WHITE, font=_FONT)
```

Define `_FONT` once at module level and pass `font=_FONT` to every `Text()` call.

### MathTex + Text combos

When a line mixes prose and math, use `VGroup` with `arrange`:

```python
label = VGroup(
    Text("Good reward ", font_size=20, color="#3fb950", font=_FONT),
    MathTex(r"(\hat{A} > 0)", font_size=28, color="#3fb950"),
    Text(": increase P(action)", font_size=20, color="#3fb950", font=_FONT),
).arrange(RIGHT, buff=0.1)
```

### atexit hook for docs copy

Every Manim script must auto-copy the final MP4 to `docs/papers/ARXIV_ID/`:

```python
import atexit, shutil
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parent.parent
_DOCS_DIR = _PAPER_DIR.parent.parent / "docs" / "papers" / "ARXIV_ID"

def _copy_to_docs():
    src = _PAPER_DIR / "output/animations/videos/NN_name/720p30/ClassName.mp4"
    dst = _DOCS_DIR / "ClassName.mp4"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")

atexit.register(_copy_to_docs)
```

### Render command

```bash
/path/to/.venv/bin/manim -qm --media_dir ../output/animations NN_name.py ClassName
```

### Color palette (matches docs dark theme)

| Role        | Hex       | Usage                         |
|-------------|-----------|-------------------------------|
| Blue        | `#58a6ff` | Section headers, highlights   |
| Green       | `#3fb950` | Positive / good / survives    |
| Orange      | `#f0883e` | Key insights, PPO connection  |
| Red         | `#E74C3C` | Negative / bad / vanishes     |
| Purple      | `#d2a8ff` | Neural net, secondary concept |
| Grey        | `GREY`    | Subtitles, context            |
| Background  | `#0d1117` | Always set camera background  |

### Scene setup

```python
class MyScene(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"
```

---

## 2. Plotly Interactive HTMLs

### The `save_plotly_html` utility

Always use `shared/plotly_utils.py`. It:
- Loads plotly.js from CDN (~50KB HTML vs ~4.7MB embedded)
- Sets `default_width='100%'` and `default_height='100%'`
- **Post-processes the HTML** to inject zero-margin CSS reset:

```css
html,body{margin:0;padding:0;overflow:hidden;background:#0d1117;width:100%;height:100%}
.plotly-graph-div{width:100%!important;height:100%!important}
```

This eliminates whitespace when rendered inside an iframe.

### Figure layout conventions

```python
fig.update_layout(
    template="plotly_dark",
    margin=dict(t=60, b=80, l=50, r=30),  # Tight margins, not t=80/b=100
    # Do NOT set height or width — let CSS handle it
    # Do NOT set autosize — the injected CSS forces 100%
)
```

**Never set `width=` in `update_layout`** — it produces a fixed-pixel div that causes
whitespace on wider screens. The CSS handles responsiveness.

**Never set `height=` in `update_layout`** — let `default_height='100%'` fill the iframe.

### Slider-driven annotations

When annotations need to update per slider step, pass them in the step's `args[1]`:

```python
steps = []
for i, val in enumerate(values):
    step = dict(
        method="update",
        args=[
            {"visible": [False] * len(fig.data)},             # traces
            {"annotations": annotations_per_step[i]},          # layout update
        ],
        label=f"{val:.0f}"
    )
    step["args"][0]["visible"][i * n_traces] = True
    steps.append(step)
```

When doing this, subplot titles (created by `make_subplots`) are replaced by the annotation
update, so you must re-include them as annotations in every step.

### Color conventions (Plotly)

Use the same hex values from the Manim palette above. Common mappings:
- Positive bars: `#27AE60` (COLORS['green'])
- Negative bars: `#E74C3C` (COLORS['red'])
- Neutral/reference: `#2E86C1` (COLORS['blue'])

---

## 3. Matplotlib Static Figures

### Style application

```python
from shared.style import apply_style, COLORS
apply_style()
```

This sets `seaborn-v0_8-whitegrid` with `#fafafa` background, consistent fonts, and grid alpha.

### Save pattern

```python
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
plt.close()
```

### PNGs are served from GitHub raw

Static PNGs are **not** stored in `docs/`. The docs page loads them from:
```
https://raw.githubusercontent.com/csking101/LuCiD-papers/main/papers/ARXIV_ID/output/static/NN_name.png
```

This keeps the `docs/` directory small. Only MP4s, interactive HTMLs, and JSONs go in `docs/`.

---

## 4. Docs Page (`index.html`)

### Page structure (single-file SPA)

Each paper page is one self-contained HTML file (~1500-2000 lines):

```
<html> (dark theme)
├── <head>: MathJax 3 CDN, inline <style>
├── <body>
│   ├── .scroll-progress-bar
│   ├── <nav> (sticky, hidden > 1400px — replaced by sidebar)
│   ├── .sidebar-toc (fixed left, shown > 1400px)
│   ├── .content
│   │   ├── .hero (title, paper link, TL;DR)
│   │   ├── .section (repeated per topic)
│   │   │   ├── prose notes
│   │   │   ├── .callout-key / .callout-insight / .callout-note
│   │   │   ├── .formal-def (purple left border, for math definitions)
│   │   │   └── .viz-section (colored left border by tool)
│   │   │       ├── <img> (static PNG from raw.githubusercontent.com)
│   │   │       ├── <iframe> (interactive HTML, same directory)
│   │   │       └── <video> (Manim MP4, same directory)
│   │   └── footer
│   ├── .modal-overlay (fullscreen iframe viewer)
│   └── <script> (scroll observer, ToC highlight, modal, back-to-top)
```

### Dark theme colors

```css
--bg:        #0d1117;
--surface:   #161b22;
--border:    #30363d;
--text:      #c9d1d9;
--heading:   #e6edf3;
--link:      #58a6ff;
```

### Iframe embedding (critical — no whitespace)

```css
.viz-section iframe {
    width: 100%;
    height: 600px;
    border: 1px solid #30363d;
    border-radius: 6px;
    background: #0d1117;     /* MUST match dark bg, never #fff */
}
```

The iframe background **must be dark** (`#0d1117`). A white background creates visible
whitespace around the Plotly chart (which has its own dark background).

The Plotly HTML inside the iframe has its own zero-margin CSS injection
(from `save_plotly_html`), so the chart fills edge-to-edge.

### Fullscreen modal

```css
.modal-overlay iframe {
    background: #0d1117;     /* Same dark bg rule */
}
```

### Responsive breakpoints

```css
@media (max-width: 1400px)  /* Hide sidebar, show nav */
@media (max-width: 768px)   /* Stack layouts, smaller fonts */
@media (max-width: 480px)   /* Minimum viable mobile */
```

### Viz section left-border colors

```css
.viz-section[data-tool="manim"]       { border-left-color: #3fb950; }
.viz-section[data-tool="matplotlib"]  { border-left-color: #58a6ff; }
.viz-section[data-tool="plotly"]      { border-left-color: #f0883e; }
.viz-section[data-tool="multi"]       { border-left-color: #d2a8ff; }
```

### MathJax

Loaded from CDN in `<head>`. Use `\( ... \)` for inline, `\[ ... \]` for display.
Configure to process `tex-chtml`:

```html
<script>
MathJax = { tex: { inlineMath: [['\\(','\\)']] } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
```

### Footer

```
Notes by Chinmaya Sahu. Content polish, code, visualizations, and web design by AI.
```

---

## 5. Landing Page (`docs/index.html`)

### Paper card

Each paper is a `.paper-card` with:
- Status badge (`.badge-done`, `.badge-learning`, or `.upcoming` overlay)
- Title + one-liner
- Viz tags (e.g., "5 PNGs", "6 HTMLs", "5 Animations")
- Link to `papers/ARXIV_ID/`

When a paper moves to Done: remove `.upcoming` class, change badge from
`badge-learning` to `badge-done`, add viz count tags, add href.

---

## 6. Git & Deployment

### LFS rules

- `*.mp4` tracked by LFS globally (`.gitattributes`)
- `docs/**/*.mp4 filter= diff= merge= -text` — exempted from LFS
  (GitHub Pages can't serve LFS pointers)
- Manim build artifacts (`partial_movie_files/`, `texts/`) are gitignored

### What goes where

| File type          | Location                          | In git?         |
|--------------------|------------------------------------|-----------------|
| Script source      | `papers/ARXIV_ID/scripts/`        | Yes             |
| Static PNG         | `papers/ARXIV_ID/output/static/`  | Yes             |
| Interactive HTML   | `papers/ARXIV_ID/output/interactive/` + `docs/papers/ARXIV_ID/` | Yes (both) |
| Manim build files  | `papers/ARXIV_ID/output/animations/` | Gitignored   |
| Manim final MP4    | `docs/papers/ARXIV_ID/`           | Yes (not LFS)   |
| Docs page          | `docs/papers/ARXIV_ID/index.html` | Yes             |

### Deployment

Push to `main`. GitHub Pages auto-deploys from `/docs` directory.

---

## 7. Python Environment

### Virtual env

All scripts run with `.venv/bin/python` (or absolute path when workdir differs).
Key packages: `matplotlib 3.10+`, `plotly 6+`, `numpy 2+`, `manim 0.20+`.

### NumPy 2.x

System matplotlib may be incompatible with NumPy 2. Always use the venv.

### PyTorch (implementations only)

Force `DEVICE = "cpu"` for small-scale implementations. CUDA kernel-launch
overhead on tiny tensors causes catastrophic slowdowns.

---

## 8. Common Pitfalls (Lessons Learned)

| Pitfall | Fix |
|---------|-----|
| White border around Plotly in iframe | Set iframe `background: #0d1117`, inject zero-margin CSS in HTML |
| Plotly chart doesn't fill iframe | Remove `height=`/`width=` from layout, use `default_height='100%'` |
| MathTex renders but SVG conversion fails | Usually a cached partial. Re-run the render. Check `dvisvgm >= 2.4` |
| Manim `Text()` looks pixelated/ugly | Switch to `font="Latin Modern Roman"` to match LaTeX aesthetic |
| Plotly slider doesn't update annotations | Pass annotations in `args[1]` of each step (layout update dict) |
| Subplot titles vanish after slider change | Re-include them as manual annotations in every slider step |
| PNGs missing on GitHub Pages | They load from `raw.githubusercontent.com`, not `docs/`. Push source output. |
| MP4s fail on GitHub Pages | Ensure `docs/**/*.mp4` is exempted from LFS in `.gitattributes` |
| Large Plotly HTML files (>4MB) | Always use `include_plotlyjs='cdn'` via `save_plotly_html()` |
| `manim` binary not found from different workdir | Use absolute path: `/path/to/.venv/bin/manim` |
