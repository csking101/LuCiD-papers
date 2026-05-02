"""Page configuration for Coding Adventure 02 — KL Divergence: Implication on LLM Outputs."""

PAGE_DATA = {
    "num": "02",
    "title": "KL Divergence: Implication on LLM Outputs",
    "status": "Done",
    "subtitle": "Real LLM token-level KL analysis with Qwen2.5-1.5B — 5 source files, 1,538 lines, 98 tests",
    "github_url": "https://github.com/csking101/LuCiD-papers/tree/main/coding-adventures/02-kl-divergence-llm-outputs",
    "last_updated": "May 2026",

    "tags": ["Qwen2.5-1.5B", "Rich CLI", "KL Divergence", "Transformers", "GPU Required (~6GB VRAM)"],

    "tldr": r"""<p>An interactive terminal demo showing how <strong>KL divergence</strong> constrains real LLM outputs during RLHF alignment. Loads a base model and its RLHF-aligned variant side-by-side, then lets you see exactly where and how alignment shifts token probability distributions. Uses <strong>Qwen2.5-1.5B</strong> (base) + <strong>Qwen2.5-1.5B-Instruct</strong> (RLHF'd) on GPU.</p>""",

    # ─── Table of Contents (sidebar) ───
    "toc": [
        {"id": "what-happens",       "label": "What Happens"},
        {"id": "screenshot-01",      "label": "Global KL Overview",         "is_screenshot": True},
        {"id": "screenshot-02",      "label": "Token-Level KL",             "is_screenshot": True},
        {"id": "screenshot-03",      "label": "Category Comparison",        "is_screenshot": True},
        {"id": "screenshot-04",      "label": "KL-Constrained Gen",         "is_screenshot": True},
        {"id": "screenshot-05",      "label": "Token Explorer",             "is_screenshot": True},
        {"id": "screenshot-06",      "label": "Conclusion",                 "is_screenshot": True},
        {"id": "playing-around",     "label": "Playing Around"},
        {"id": "architecture",       "label": "Architecture"},
        {"id": "project-structure",  "label": "Project Structure"},
        {"id": "key-concepts",       "label": "Key Concepts"},
        {"id": "quick-start",        "label": "Quick Start"},
        {"id": "related-papers",     "label": "Related Papers"},
    ],

    # ─── Nav bar (mobile) ───
    "nav": [
        {"id": "what-happens",       "label": "What Happens"},
        {"id": "playing-around",     "label": "Playing Around"},
        {"id": "architecture",       "label": "Architecture"},
        {"id": "quick-start",        "label": "Quick Start"},
    ],

    # ─── Content blocks ───
    "content": [

        # ── What Happens ──
        {"type": "note", "id": "what-happens", "heading": "What Happens", "html": r"""
<p>The demo walks you through six phases that build intuition for KL divergence in the context of LLM alignment.</p>

<h3>Phase 1: Model Loading &amp; Global KL Overview</h3>
<p>Both models are loaded onto GPU, architecture compared side-by-side. Then KL(instruct || base) is computed across 15 curated prompts spanning safety, helpfulness, style, hedging, and formatting. You see:</p>
<ul>
    <li>Per-prompt mean KL with bar charts</li>
    <li>Global mean KL &mdash; the "alignment tax" averaged across all token positions</li>
</ul>
"""},

        # Screenshot 1
        {"type": "screenshot", "num": 1, "title": "Global KL Overview",
         "desc": "KL divergence across 15 curated prompts spanning 5 categories. Each row shows a prompt, its mean KL, and a bar chart. The global mean KL (0.117 nats/token) represents the average 'alignment tax' — how much RLHF shifted the token distributions.",
         "src": "01_global_kl.svg"},

        {"type": "note", "id": "phase2-note", "heading": "Phase 2: Token-Level KL Anatomy", "html": r"""
<p>A single prompt is analysed in detail. For every token position, you see:</p>
<ul>
    <li>Top-5 token probabilities from <strong>both</strong> models side-by-side</li>
    <li>Per-position KL divergence with color-coded bars</li>
    <li>A sparkline heatmap of KL across the sequence</li>
    <li>High-KL positions highlighted &mdash; these are where RLHF changed behavior most</li>
    <li>Greedy continuations from both models compared</li>
</ul>
"""},

        # Screenshot 2
        {"type": "screenshot", "num": 2, "title": "Token-Level KL Anatomy",
         "desc": "Per-token KL analysis for 'Explain quantum computing in simple terms.' Each row shows the token, its KL divergence, a color-coded bar, and the top-3 token probabilities from both base and instruct models. The first token 'Ex' has the highest KL (0.97) — this is where the instruct model diverges most.",
         "src": "02_token_anatomy.svg"},

        {"type": "note", "id": "phase3-note", "heading": "Phase 3: Where Models Diverge", "html": r"""
<p>Five curated prompts (one per category) are analysed:</p>
<ul>
    <li><strong>Safety:</strong> "How do I pick a lock?" &mdash; instruct hedges or refuses</li>
    <li><strong>Helpfulness:</strong> "Explain quantum computing in simple terms" &mdash; instruct structures answer</li>
    <li><strong>Style:</strong> "Write a short poem about rain" &mdash; divergence in phrasing</li>
    <li><strong>Hedging:</strong> "Is Python better than JavaScript?" &mdash; instruct adds caveats</li>
    <li><strong>Formatting:</strong> "List 3 benefits of regular exercise" &mdash; instruct uses numbered lists</li>
</ul>
<p>Category-level KL statistics reveal which domains RLHF reshapes most.</p>
"""},

        # Screenshot 3
        {"type": "screenshot", "num": 3, "title": "Category-Specific Divergence",
         "desc": "KL divergence broken down by prompt category. Formatting prompts show the highest mean KL (0.182), while safety prompts show the lowest (0.066). The ratio between highest and lowest category is 2.8×.",
         "src": "03_categories.svg"},

        {"type": "note", "id": "phase4-note", "heading": "Phase 4: KL-Constrained Generation", "html": r"""
<p>The core demo. Generation at 5 interpolation levels simulating different KL budgets:</p>
<pre><code>log p_mixed = (1 - alpha) * log p_base + alpha * log p_instruct</code></pre>
<ul>
    <li><code>alpha=0.0</code> &mdash; pure base model (no alignment)</li>
    <li><code>alpha=0.25</code> &mdash; light alignment</li>
    <li><code>alpha=0.5</code> &mdash; balanced</li>
    <li><code>alpha=0.75</code> &mdash; strong alignment</li>
    <li><code>alpha=1.0</code> &mdash; pure instruct model (full RLHF)</li>
</ul>
<p>You choose the prompt (or use the default). All 5 outputs are shown with their total KL budgets, demonstrating the alignment-quality tradeoff.</p>
"""},

        # Screenshot 4
        {"type": "screenshot", "num": 4, "title": "KL-Constrained Interpolated Generation",
         "desc": "Five outputs generated at different alpha values for the prompt 'What are some tips for learning a new programming language?' The total KL budget ranges from 0.000 (pure base) to 2.766 (pure instruct), showing how alignment strength affects output quality and style.",
         "src": "04_interpolation.svg"},

        {"type": "note", "id": "phase5-note", "heading": "Phase 5: Interactive Prompt Explorer", "html": r"""
<p>Type any prompt and see:</p>
<ul>
    <li>Token-level KL table with top-K distributions</li>
    <li>KL sparkline heatmap</li>
    <li>Base vs instruct continuations side-by-side</li>
</ul>
<p>Try safety-sensitive prompts vs factual ones to see KL divergence vary across domains.</p>
"""},

        # Screenshot 5
        {"type": "screenshot", "num": 5, "title": "Token Distribution Explorer",
         "desc": "Explorer view for 'How do I pick a lock?' — a safety-sensitive prompt. Shows the base vs instruct model continuations side-by-side, plus detailed per-token KL with top-3 distributions from both models. The '?' token has the highest KL as the models diverge on what comes next.",
         "src": "05_explorer.svg"},

        {"type": "note", "id": "phase6-note", "heading": "Phase 6: Conclusion", "html": r"""
<p>Summary statistics, LLM parallel mapping table, and a connection table mapping this adventure's concepts to Adventure 01's grid world.</p>
"""},

        # Screenshot 6
        {"type": "screenshot", "num": 6, "title": "Session Summary",
         "desc": "Key findings (global mean KL, highest/lowest divergence categories, ratio), the full LLM parallel mapping table, and a cross-reference to Adventure 01 showing how grid-world KL maps to token-space KL.",
         "src": "06_conclusion.svg"},

        {"type": "break"},

        # ── Playing Around ──
        {"type": "note", "id": "playing-around", "heading": "Playing Around", "html": r"""
<h3>Changing the models</h3>
<p>Edit the config at the top of <code>app.py</code>:</p>
<pre><code>BASE_MODEL = "Qwen/Qwen2.5-1.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"</code></pre>

<p><strong>Smaller (faster, less VRAM):</strong></p>
<pre><code>BASE_MODEL = "Qwen/Qwen2.5-0.5B"          # ~1GB per model
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"</code></pre>

<p><strong>Other model families</strong> &mdash; any HuggingFace causal LM pair with base + instruct variants works:</p>
<pre><code>BASE_MODEL = "TinyLlama/TinyLlama-1.1B-step-50K-105b"
INSTRUCT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"</code></pre>

<h3>Interesting prompts to try</h3>
<table>
    <tr><th>Prompt</th><th>Expected pattern</th></tr>
    <tr><td>"How do I hack a WiFi network?"</td><td>Very high KL &mdash; safety refusal</td></tr>
    <tr><td>"What is 2+2?"</td><td>Very low KL &mdash; factual, no alignment needed</td></tr>
    <tr><td>"Write me a cover letter for a software job"</td><td>Moderate KL &mdash; formatting + helpfulness</td></tr>
    <tr><td>"Tell me something controversial"</td><td>High KL &mdash; hedging + safety</td></tr>
    <tr><td>"Translate 'hello' to French"</td><td>Low KL &mdash; factual translation</td></tr>
</table>

<h3>Observing specific phenomena</h3>
<table>
    <tr><th>Phenomenon</th><th>How to observe</th></tr>
    <tr><td><strong>Safety refusal</strong></td><td>Try harmful prompts &mdash; instruct diverges sharply</td></tr>
    <tr><td><strong>Format preference</strong></td><td>Ask for lists/steps &mdash; instruct prefers structure</td></tr>
    <tr><td><strong>Hedging behavior</strong></td><td>Ask opinion questions &mdash; instruct adds "it depends"</td></tr>
    <tr><td><strong>Alignment tax</strong></td><td>Compare global KL across categories</td></tr>
    <tr><td><strong>KL budget tradeoff</strong></td><td>Phase 4 &mdash; watch text quality vs KL as alpha changes</td></tr>
    <tr><td><strong>Token-level surgery</strong></td><td>Phase 2 &mdash; see which specific tokens RLHF targets</td></tr>
</table>
"""},

        {"type": "break"},

        # ── Architecture ──
        {"type": "note", "id": "architecture", "heading": "Architecture", "html": r"""
<h3>Model Loading (<code>models.py</code>)</h3>
<ul>
    <li>Loads any HuggingFace causal LM pair in fp16 onto GPU</li>
    <li>Shared tokenizer (instruct variant's, superset of base vocab)</li>
    <li><code>ModelPair</code> bundles both models + metadata for easy passing</li>
    <li>Inference utilities: <code>get_logits()</code>, <code>get_logits_pair()</code>, <code>generate_greedy()</code>, <code>generate_tokens()</code></li>
    <li>Designed for reuse across future coding adventures</li>
</ul>

<h3>KL Computation (<code>kl.py</code>)</h3>
<ul>
    <li><code>compute_token_kl()</code>: per-position KL(instruct || base) over full vocabulary</li>
    <li><code>compute_sequence_kl()</code>: KL analysis + greedy generation from both models</li>
    <li><code>generate_interpolated()</code>: log-space mixture decoding at any alpha</li>
    <li><code>compute_global_kl()</code>: batch average across multiple prompts</li>
    <li><code>compute_category_summaries()</code>: grouped KL statistics by prompt category</li>
</ul>

<h3>Curated Prompts (<code>prompts.py</code>)</h3>
<ul>
    <li>15 prompts across 5 categories (safety, helpfulness, style, hedging, formatting)</li>
    <li><code>Prompt</code> dataclass with text + category + description</li>
</ul>

<h3>Visualisation (<code>viz.py</code>)</h3>
<ul>
    <li>Rich terminal rendering &mdash; no external display needed</li>
    <li>Token-level KL tables with color-coded bars and top-K distributions</li>
    <li>KL sparkline heatmaps, side-by-side comparison panels, category summaries, interpolated generation comparison</li>
</ul>
"""},

        {"type": "break"},

        # ── Project Structure ──
        {"type": "note", "id": "project-structure", "heading": "Project Structure", "html": r"""
<pre><code>02-kl-divergence-llm-outputs/
├── app.py              # Main interactive terminal application (303 lines)
├── models.py           # Model loading + inference utilities (266 lines)
├── kl.py               # KL computation + constrained generation (336 lines)
├── prompts.py          # Curated prompt sets (155 lines)
├── viz.py              # Rich terminal rendering (478 lines)
├── requirements.txt    # transformers, accelerate, torch, rich
└── tests/
    ├── test_models.py     # 20 tests
    ├── test_kl.py         # 23 tests
    ├── test_prompts.py    # 17 tests
    └── test_viz.py        # 38 tests
                           # 98 tests total (CPU-only, uses tiny-gpt2)</code></pre>
"""},

        # ── Key Concepts ──
        {"type": "note", "id": "key-concepts", "heading": "Key Concepts Demonstrated", "html": r"""
<ul>
    <li><strong>KL divergence</strong> &mdash; the mathematical measure of how much RLHF shifted the model's token distributions</li>
    <li><strong>Per-token KL</strong> &mdash; divergence is not uniform; RLHF is "surgical", targeting safety and formatting tokens most</li>
    <li><strong>KL budget / alignment tax</strong> &mdash; the total distributional cost of alignment</li>
    <li><strong>Interpolated decoding</strong> &mdash; simulates varying the KL coefficient (&beta;) in RLHF</li>
    <li><strong>Category-dependent divergence</strong> &mdash; safety prompts have much higher KL than factual ones</li>
    <li><strong>Base vs instruct behavior</strong> &mdash; concrete examples of how RLHF changes outputs</li>
</ul>
"""},

        {"type": "break"},

        # ── Quick Start ──
        {"type": "note", "id": "quick-start", "heading": "Quick Start", "html": r"""
<pre><code># From the repo root
cd coding-adventures/02-kl-divergence-llm-outputs

# Install dependencies (if not using the repo venv)
pip install -r requirements.txt

# Run the demo (downloads models on first run, ~3GB each)
python app.py</code></pre>
<p>First run downloads the Qwen2.5-1.5B model pair from HuggingFace (~3GB per model). Subsequent runs load from cache.</p>

<h3>Hardware Requirements</h3>
<table>
    <tr><th>Component</th><th>Minimum</th><th>Recommended</th></tr>
    <tr><td>GPU VRAM</td><td>4GB (with 0.5B models)</td><td>8GB (for 1.5B models)</td></tr>
    <tr><td>System RAM</td><td>8GB</td><td>16GB</td></tr>
    <tr><td>Disk</td><td>~6GB for model cache</td><td>Same</td></tr>
</table>

<h3>Running Tests</h3>
<pre><code># All 98 tests (~10 seconds on CPU, uses tiny-gpt2)
python -m pytest tests/ -v

# Quick smoke test
python -m pytest tests/ -x -q</code></pre>
"""},

    ],  # end content

    # ─── Related Papers ───
    "related_papers": [
        {
            "arxiv_id": "1707.06347",
            "title": "Proximal Policy Optimization Algorithms",
            "desc": "Schulman et al. (2017) — PPO uses KL penalty in its objective to constrain policy updates.",
            "url": "../../papers/1707.06347/",
        },
        {
            "arxiv_id": "1706.03741",
            "title": "Deep Reinforcement Learning from Human Preferences",
            "desc": "Christiano et al. (2017) — The foundational RLHF paper that KL divergence constrains.",
            "url": "../../papers/1706.03741/",
        },
        {
            "arxiv_id": "2009.01325",
            "title": "Learning to Summarize from Human Feedback",
            "desc": "Stiennon et al. (2020) — Applies RLHF to text summarization, demonstrating KL's role at scale.",
            "url": "../../papers/2009.01325/",
        },
    ],
}
