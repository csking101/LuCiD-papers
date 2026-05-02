"""Page configuration for Coding Adventure 04 — System Prompt Steering."""

PAGE_DATA = {
    "num": "04",
    "title": "System Prompt Steering",
    "status": "Done",
    "subtitle": "How system prompts steer token distributions — 5 source files, ~1,600 lines, 128 tests",
    "github_url": "https://github.com/csking101/LuCiD-papers/tree/main/coding-adventures/04-system-prompt-steering",
    "last_updated": "May 2026",

    "tags": ["Qwen2.5-1.5B-Instruct", "Rich CLI", "KL Divergence", "System Prompts", "GPU Required (~3GB VRAM)"],

    "tldr": r"""<p>An interactive terminal demo measuring how much <strong>system prompts</strong> steer a single instruct model's token distributions. Loads <strong>Qwen2.5-1.5B-Instruct</strong> and compares first-token probability distributions under 7 different system prompts (safety, persona, formatting, expert, brevity, opposing) across 7 user prompts. Shows that formatting/persona prompts are 36x stronger steerers than safety prompts.</p>""",

    # --- Table of Contents (sidebar) ---
    "toc": [
        {"id": "what-happens",       "label": "What Happens"},
        {"id": "screenshot-01",      "label": "Chat Template",              "is_screenshot": True},
        {"id": "screenshot-02",      "label": "First-Token Steering",       "is_screenshot": True},
        {"id": "screenshot-03",      "label": "Forced Continuation",        "is_screenshot": True},
        {"id": "screenshot-04",      "label": "Steering Matrix",            "is_screenshot": True},
        {"id": "screenshot-05",      "label": "Explorer",                   "is_screenshot": True},
        {"id": "screenshot-06",      "label": "Conclusion",                 "is_screenshot": True},
        {"id": "playing-around",     "label": "Playing Around"},
        {"id": "architecture",       "label": "Architecture"},
        {"id": "project-structure",  "label": "Project Structure"},
        {"id": "key-concepts",       "label": "Key Concepts"},
        {"id": "quick-start",        "label": "Quick Start"},
        {"id": "related-papers",     "label": "Related Papers"},
    ],

    # --- Nav bar (mobile) ---
    "nav": [
        {"id": "what-happens",       "label": "What Happens"},
        {"id": "playing-around",     "label": "Playing Around"},
        {"id": "architecture",       "label": "Architecture"},
        {"id": "quick-start",        "label": "Quick Start"},
    ],

    # --- Content blocks ---
    "content": [

        # -- What Happens --
        {"type": "note", "id": "what-happens", "heading": "What Happens", "html": r"""
<p>The demo walks you through six phases measuring how system prompts steer an instruct model's token distributions.</p>

<h3>Phase 1: Model &amp; Chat Template</h3>
<p>The model is loaded onto GPU. You see the chat template format &mdash; how Qwen wraps system + user messages with special tokens. Different system prompts change the prefix tokens, shifting the model's hidden state before it generates the first response token.</p>
"""},

        # Screenshot 1
        {"type": "screenshot", "num": 1, "title": "Chat Template Comparison",
         "desc": "Side-by-side view of how Qwen's chat template formats messages with the Default system prompt vs a Safety system prompt. The special tokens (<|im_start|>, <|im_end|>) structure the conversation, and the system prompt content changes the model's conditioning.",
         "src": "01_chat_template.svg"},

        {"type": "note", "id": "phase2-note", "heading": "Phase 2: First-Token Steering", "html": r"""
<p>A single user prompt is analysed with two system prompts. For the first generated token, you see:</p>
<ul>
    <li>Top-K token probabilities from <strong>both</strong> conditions side-by-side</li>
    <li>KL(Custom || Default) and Jensen-Shannon divergence</li>
    <li>Biggest probability shifts &mdash; which tokens gained or lost probability</li>
</ul>
"""},

        # Screenshot 2
        {"type": "screenshot", "num": 2, "title": "First-Token Distribution Comparison",
         "desc": "First-token analysis for 'How do I pick a lock?' with Default vs Safety system prompt. KL divergence is 0.176 nats. The biggest shift: 'Sorry' drops from 28.7% to 9.6% and 'I' rises from 56.2% to 71.8% under the Safety prompt.",
         "src": "02_first_token.svg"},

        {"type": "note", "id": "phase3-note", "heading": "Phase 3: Forced-Continuation KL Profile", "html": r"""
<p>Generates a continuation from the Default system prompt, then forces those exact same tokens through the Pirate persona prompt. The per-position KL shows where the model 'fights' against the forced tokens:</p>
<ul>
    <li>Position 0 ('Quant') has 10.3 KL &mdash; the Pirate prompt strongly prefers a different opening</li>
    <li>Content tokens ('quantum', 'mechanics') have near-zero KL &mdash; factual content is unaffected</li>
    <li>Transition tokens ('is', 'a', 'type') have high KL &mdash; the Pirate prompt wants different phrasing</li>
</ul>
"""},

        # Screenshot 3
        {"type": "screenshot", "num": 3, "title": "Forced-Continuation KL Profile",
         "desc": "Per-token KL when forcing Default's continuation under the Pirate prompt. The first token has 10.3 KL — the Pirate persona strongly disagrees with the formal opening. Side-by-side: Default says 'Quantum computing is a type of computing...' while Pirate says 'Ahoy there! Quantum computing is like having a ship...'",
         "src": "03_forced_continuation.svg"},

        {"type": "note", "id": "phase4-note", "heading": "Phase 4: Steering Matrix", "html": r"""
<p>The core analysis: 6 system prompts &times; 7 user prompts = 42 first-token KL comparisons. Results show:</p>
<ul>
    <li><strong>Bullet Points</strong> and <strong>Pirate</strong> are the strongest steerers (~11 KL mean)</li>
    <li><strong>Safety</strong> is the weakest steerer (~0.3 KL) &mdash; the model already has safety training</li>
    <li>The ratio between strongest and weakest is <strong>36.7&times;</strong></li>
    <li>Format-forcing prompts steer more than content-forcing prompts</li>
</ul>
"""},

        # Screenshot 4
        {"type": "screenshot", "num": 4, "title": "Steering Matrix",
         "desc": "Full 6x7 steering matrix showing KL(Custom || Default) for each system prompt x user prompt combination. Pirate and Bullet Points dominate (>7 KL everywhere), while Safety barely registers (<1 KL). The profile table ranks system prompts by mean steering power.",
         "src": "04_steering_matrix.svg"},

        {"type": "note", "id": "phase5-note", "heading": "Phase 5: Interactive Explorer", "html": r"""
<p>Type any system prompt + user prompt and see:</p>
<ul>
    <li>First-token distribution comparison with KL and JS divergence</li>
    <li>Top-K probability shifts</li>
    <li>Side-by-side generated outputs (Default vs Custom)</li>
</ul>
"""},

        # Screenshot 5
        {"type": "screenshot", "num": 5, "title": "Interactive Explorer",
         "desc": "Explorer view: Pirate persona on 'How do I pick a lock?' — KL jumps to 11.3 nats. Default starts with 'I' (56.2%) while Pirate starts with 'Ah' (67.7%). The entire top-K vocabulary shifts from formal to pirate speak.",
         "src": "05_explorer.svg"},

        {"type": "note", "id": "phase6-note", "heading": "Phase 6: Conclusion", "html": r"""
<p>Summary statistics, LLM parallel mapping table (system prompts as soft constraints vs KL penalty as hard constraints), and a three-way connection table mapping concepts across Adventures 01, 02, and 04.</p>
"""},

        # Screenshot 6
        {"type": "screenshot", "num": 6, "title": "Session Summary",
         "desc": "Key findings (global mean KL 4.25 nats, strongest/weakest steerers, 36.7x ratio), the LLM parallel mapping, and cross-adventure connections showing how grid-world KL, base-vs-instruct KL, and system-prompt KL relate.",
         "src": "06_conclusion.svg"},

        {"type": "break"},

        # -- Playing Around --
        {"type": "note", "id": "playing-around", "heading": "Playing Around", "html": r"""
<h3>Adding your own system prompts</h3>
<p>Edit <code>prompts.py</code> to add new system prompts:</p>
<pre><code>MY_PROMPT = SystemPrompt(
    name="Storyteller",
    text="Tell everything as a fairy tale story.",
    category="persona",
    description="Forces narrative storytelling style",
)</code></pre>

<h3>Interesting experiments</h3>
<table>
    <tr><th>Experiment</th><th>What to observe</th></tr>
    <tr><td>Pirate + "What is 2+2?"</td><td>Huge KL even for trivial facts — persona dominates</td></tr>
    <tr><td>Safety + "How to bake cookies"</td><td>Near-zero KL — safety prompt is irrelevant here</td></tr>
    <tr><td>One Sentence + formatting questions</td><td>Conflict between brevity and list expectations</td></tr>
    <tr><td>Devil's Advocate + opinions</td><td>Strongest where there's a clear premise to oppose</td></tr>
</table>

<h3>Using a different model</h3>
<p>Edit <code>app.py</code>:</p>
<pre><code>MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # smaller, faster</code></pre>
<p>Any HuggingFace instruct model with a chat template should work.</p>
"""},

        {"type": "break"},

        # -- Architecture --
        {"type": "note", "id": "architecture", "heading": "Architecture", "html": r"""
<h3>Model Loading (<code>models.py</code>)</h3>
<ul>
    <li>Loads a single HuggingFace instruct model in fp16 onto GPU</li>
    <li>Chat template formatting via <code>tokenizer.apply_chat_template()</code></li>
    <li><code>get_first_token_dist()</code> extracts the distribution over the first generated token</li>
    <li>Greedy generation for side-by-side output comparison</li>
</ul>

<h3>Steering Analysis (<code>analysis.py</code>)</h3>
<ul>
    <li><code>compare_first_token()</code>: KL and JS divergence between two system prompt conditions</li>
    <li><code>compute_forced_continuation_kl()</code>: per-position KL when forcing one condition's output under another</li>
    <li><code>compute_steering_matrix()</code>: batch first-token KL for system_prompt &times; user_prompt grid</li>
    <li><code>compute_system_prompt_profiles()</code>: aggregated steering power per system prompt</li>
</ul>

<h3>Curated Prompts (<code>prompts.py</code>)</h3>
<ul>
    <li>7 system prompts across 7 categories (baseline, safety, persona, formatting, expert, brevity, opposing)</li>
    <li>7 user prompts across 7 categories (safety, factual, creative, opinion, formatting, trivial, humor)</li>
</ul>

<h3>Visualisation (<code>viz.py</code>)</h3>
<ul>
    <li>Rich terminal rendering &mdash; no external display needed</li>
    <li>Chat template comparison, first-token distribution tables, forced-continuation KL profiles</li>
    <li>Steering matrix heatmap, profile rankings, adventure connection tables</li>
</ul>
"""},

        {"type": "break"},

        # -- Project Structure --
        {"type": "note", "id": "project-structure", "heading": "Project Structure", "html": r"""
<pre><code>04-system-prompt-steering/
├── app.py              # Main interactive terminal application (~340 lines)
├── models.py           # Model loading + inference utilities (~310 lines)
├── analysis.py         # Steering analysis: KL, JS, matrix (~320 lines)
├── prompts.py          # Curated system + user prompts (~170 lines)
├── viz.py              # Rich terminal rendering (~460 lines)
├── requirements.txt    # transformers, accelerate, torch, rich
└── tests/
    ├── test_models.py     # 26 tests
    ├── test_analysis.py   # 27 tests
    ├── test_prompts.py    # 36 tests
    └── test_viz.py        # 39 tests
                           # 128 tests total (CPU-only, uses tiny-gpt2)</code></pre>
"""},

        # -- Key Concepts --
        {"type": "note", "id": "key-concepts", "heading": "Key Concepts Demonstrated", "html": r"""
<ul>
    <li><strong>System prompt steering</strong> &mdash; system prompts are "soft constraints" that change the model's input context, not its weights</li>
    <li><strong>First-token KL</strong> &mdash; the distribution over the first generated token captures the system prompt's overall steering power</li>
    <li><strong>Forced-continuation KL</strong> &mdash; shows where in a sequence the system prompt exerts the most "pressure" to change the output</li>
    <li><strong>Steering matrix</strong> &mdash; reveals interaction effects between system prompt type and user prompt type</li>
    <li><strong>Soft vs hard constraints</strong> &mdash; system prompts (input conditioning) vs RLHF (weight modification) as two approaches to alignment</li>
    <li><strong>Format > content</strong> &mdash; formatting/persona prompts steer much more than safety/expert prompts (36.7x ratio)</li>
</ul>
"""},

        {"type": "break"},

        # -- Quick Start --
        {"type": "note", "id": "quick-start", "heading": "Quick Start", "html": r"""
<pre><code># From the repo root
cd coding-adventures/04-system-prompt-steering

# Install dependencies (if not using the repo venv)
pip install -r requirements.txt

# Run the demo (downloads model on first run, ~3GB)
python app.py</code></pre>
<p>First run downloads Qwen2.5-1.5B-Instruct from HuggingFace (~3GB). Subsequent runs load from cache.</p>

<h3>Hardware Requirements</h3>
<table>
    <tr><th>Component</th><th>Minimum</th><th>Recommended</th></tr>
    <tr><td>GPU VRAM</td><td>2GB (with 0.5B model)</td><td>4GB (for 1.5B model)</td></tr>
    <tr><td>System RAM</td><td>8GB</td><td>16GB</td></tr>
    <tr><td>Disk</td><td>~3GB for model cache</td><td>Same</td></tr>
</table>

<h3>Running Tests</h3>
<pre><code># All 128 tests (~14 seconds on CPU, uses tiny-gpt2)
python -m pytest tests/ -v

# Quick smoke test
python -m pytest tests/ -x -q</code></pre>
"""},

    ],  # end content

    # --- Related Papers ---
    "related_papers": [
        {
            "arxiv_id": "2203.02155",
            "title": "Training language models to follow instructions with human feedback (InstructGPT)",
            "desc": "Ouyang et al. (2022) — System prompts build on the instruction-following capability that InstructGPT pioneered.",
            "url": "../../papers/2203.02155/",
        },
        {
            "arxiv_id": "1707.06347",
            "title": "Proximal Policy Optimization Algorithms",
            "desc": "Schulman et al. (2017) — PPO's KL penalty is the 'hard constraint' counterpart to system prompts' 'soft constraint'.",
            "url": "../../papers/1707.06347/",
        },
        {
            "arxiv_id": "1706.03741",
            "title": "Deep Reinforcement Learning from Human Preferences",
            "desc": "Christiano et al. (2017) — The foundational RLHF paper. System prompts steer without retraining.",
            "url": "../../papers/1706.03741/",
        },
    ],
}
