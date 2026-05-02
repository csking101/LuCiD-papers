# KL Divergence: Implication on LLM Outputs

An interactive terminal demo showing how **KL divergence** constrains real LLM outputs during RLHF alignment. Loads a base model and its RLHF-aligned variant side-by-side, then lets you see exactly where and how alignment shifts token probability distributions.

Uses **Qwen2.5-1.5B** (base) + **Qwen2.5-1.5B-Instruct** (RLHF'd) on GPU. Requires ~6GB VRAM.

---

## Quick Start

```bash
# From the repo root
cd coding-adventures/02-kl-divergence-llm-outputs

# Install dependencies (if not using the repo venv)
pip install -r requirements.txt

# Run the demo (downloads models on first run, ~3GB each)
python app.py
```

First run downloads the Qwen2.5-1.5B model pair from HuggingFace (~3GB per model). Subsequent runs load from cache.

---

## What Happens

The demo walks you through six phases that build intuition for KL divergence in the context of LLM alignment.

### Phase 1: Model Loading & Global KL Overview

Both models are loaded onto GPU, architecture compared side-by-side. Then KL(instruct || base) is computed across 15 curated prompts spanning safety, helpfulness, style, hedging, and formatting. You see:
- Per-prompt mean KL with bar charts
- Global mean KL -- the "alignment tax" averaged across all token positions

### Phase 2: Token-Level KL Anatomy

A single prompt is analysed in detail. For every token position, you see:
- Top-5 token probabilities from **both** models side-by-side
- Per-position KL divergence with color-coded bars
- A sparkline heatmap of KL across the sequence
- High-KL positions highlighted -- these are where RLHF changed behavior most
- Greedy continuations from both models compared

### Phase 3: Where Models Diverge

Five curated prompts (one per category) are analysed:
- **Safety**: "How do I pick a lock?" -- instruct hedges/refuses
- **Helpfulness**: "Explain quantum computing in simple terms" -- instruct structures answer
- **Style**: "Write a short poem about rain" -- divergence in phrasing
- **Hedging**: "Is Python better than JavaScript?" -- instruct adds caveats
- **Formatting**: "List 3 benefits of regular exercise" -- instruct uses numbered lists

Category-level KL statistics reveal which domains RLHF reshapes most.

### Phase 4: KL-Constrained Generation

The core demo. Generation at 5 interpolation levels simulating different KL budgets:

```
log p_mixed = (1 - alpha) * log p_base + alpha * log p_instruct
```

- `alpha=0.0` -- pure base model (no alignment)
- `alpha=0.25` -- light alignment
- `alpha=0.5` -- balanced
- `alpha=0.75` -- strong alignment
- `alpha=1.0` -- pure instruct model (full RLHF)

You choose the prompt (or use the default). All 5 outputs are shown with their total KL budgets, demonstrating the alignment-quality tradeoff.

### Phase 5: Interactive Prompt Explorer

Type any prompt and see:
- Token-level KL table with top-K distributions
- KL sparkline heatmap
- Base vs instruct continuations side-by-side

Try safety-sensitive prompts vs factual ones to see KL divergence vary across domains.

### Phase 6: Conclusion

Summary statistics, LLM parallel mapping table, and connection to Adventure 01.

---

## Playing Around

### Changing the models

Edit the config at the top of `app.py`:

```python
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

**Smaller (faster, less VRAM):**
```python
BASE_MODEL = "Qwen/Qwen2.5-0.5B"          # ~1GB per model
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
```

**Other model families** -- any HuggingFace causal LM pair with base + instruct variants works:
```python
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-step-50K-105b"
INSTRUCT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Tuning generation

```python
MAX_NEW_TOKENS = 64                     # Longer = more tokens to compare
INTERPOLATION_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]  # Add more granularity
```

### Adding custom prompts

Edit `prompts.py` to add your own curated prompts to any category, or create new categories. The phase 3 analysis will automatically include them.

### Interesting prompts to try

| Prompt | Expected pattern |
|---|---|
| "How do I hack a WiFi network?" | Very high KL -- safety refusal |
| "What is 2+2?" | Very low KL -- factual, no alignment needed |
| "Write me a cover letter for a software job" | Moderate KL -- formatting + helpfulness |
| "Tell me something controversial" | High KL -- hedging + safety |
| "Translate 'hello' to French" | Low KL -- factual translation |

### Observing specific phenomena

| Phenomenon | How to observe |
|---|---|
| **Safety refusal** | Try harmful prompts -- instruct diverges sharply |
| **Format preference** | Ask for lists/steps -- instruct prefers structure |
| **Hedging behavior** | Ask opinion questions -- instruct adds "it depends" |
| **Alignment tax** | Compare global KL across categories |
| **KL budget tradeoff** | Phase 4 -- watch text quality vs KL as alpha changes |
| **Token-level surgery** | Phase 2 -- see which specific tokens RLHF targets |

---

## Architecture

### Model Loading (`models.py`)

- Loads any HuggingFace causal LM pair in fp16 onto GPU
- Shared tokenizer (instruct variant's, superset of base vocab)
- `ModelPair` bundles both models + metadata for easy passing
- `TokenLogits` wraps raw logits + derived log-probs + probs
- Inference utilities: `get_logits()`, `get_logits_pair()`, `generate_greedy()`, `generate_tokens()`
- Designed for reuse across future coding adventures

### KL Computation (`kl.py`)

- `compute_token_kl()`: per-position KL(instruct || base) over full vocabulary
- `compute_sequence_kl()`: KL analysis + greedy generation from both models
- `generate_interpolated()`: log-space mixture decoding at any alpha
- `compute_global_kl()`: batch average across multiple prompts
- `compute_category_summaries()`: grouped KL statistics by prompt category

### Curated Prompts (`prompts.py`)

- 15 prompts across 5 categories (safety, helpfulness, style, hedging, formatting)
- `Prompt` dataclass with text + category + description
- Phase helpers: `get_phase2_prompt()`, `get_phase3_prompts()`

### Visualisation (`viz.py`)

- Rich terminal rendering -- no external display needed
- Token-level KL tables with color-coded bars and top-K distributions
- KL sparkline heatmaps
- Side-by-side base vs instruct comparison panels
- Category summary tables with ratio analysis
- Interpolated generation comparison
- LLM parallel mapping and Adventure 01 connection tables

---

## Project Structure

```
02-kl-divergence-llm-outputs/
├── app.py              # Main interactive terminal application (303 lines)
├── models.py           # Model loading + inference utilities (266 lines)
├── kl.py               # KL computation + constrained generation (336 lines)
├── prompts.py          # Curated prompt sets (155 lines)
├── viz.py              # Rich terminal rendering (478 lines)
├── requirements.txt    # transformers, accelerate, torch, rich
├── README.md
└── tests/
    ├── test_models.py     # 20 tests — loading, logits, generation
    ├── test_kl.py         # 23 tests — KL computation, interpolation, batching
    ├── test_prompts.py    # 17 tests — prompt structure, categories, helpers
    └── test_viz.py        # 38 tests — all Rich renderable components
                           # ─────────
                           # 98 tests total (CPU-only, uses tiny-gpt2)
```

---

## Key Concepts Demonstrated

- **KL divergence** -- the mathematical measure of how much RLHF shifted the model's token distributions
- **Per-token KL** -- divergence is not uniform; RLHF is "surgical", targeting safety and formatting tokens most
- **KL budget / alignment tax** -- the total distributional cost of alignment
- **Interpolated decoding** -- simulates varying the KL coefficient (beta) in RLHF
- **Category-dependent divergence** -- safety prompts have much higher KL than factual ones
- **Base vs instruct behavior** -- concrete examples of how RLHF changes outputs

---

## Running Tests

```bash
# All tests (uses tiny-gpt2, no GPU required)
python -m pytest tests/ -v

# Quick smoke test
python -m pytest tests/ -x -q
```

All 98 tests run in ~10 seconds on CPU using `sshleifer/tiny-gpt2`.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 4GB (with 0.5B models) | 8GB (for 1.5B models) |
| System RAM | 8GB | 16GB |
| Disk | ~6GB for model cache | Same |

---

## Paper References

This adventure implements concepts from:

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) -- [LuCiD visualisations](../../papers/1707.06347/)
- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) (Christiano et al., 2017) -- [LuCiD visualisations](../../papers/1706.03741/)
- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) (Stiennon et al., 2020) -- [LuCiD visualisations](../../papers/2009.01325/)

See also: [Adventure 01 -- Path-Finding Preference Game](../01-pathfinding-preference-game/) for the same KL divergence concept applied to grid-world navigation.
