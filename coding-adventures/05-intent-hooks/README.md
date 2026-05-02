# Intent Hooks — Classify and Deny Harmful Prompts

Use **PyTorch forward hooks** to intercept hidden states inside Qwen2.5-1.5B-Instruct, train **linear probes** at every layer to classify user intent (benign vs harmful), and build a **guardrail pipeline** that blocks harmful prompts before generation.

Built with PyTorch + Transformers + Rich. Requires GPU (~3GB VRAM).

---

## Quick Start

```bash
# From the repo root
cd coding-adventures/05-intent-hooks

# Install dependencies (if not using the repo venv)
pip install -r requirements.txt

# Run the demo (downloads model on first run, ~3GB)
python app.py
```

First run downloads Qwen2.5-1.5B-Instruct from HuggingFace (~3GB). Subsequent runs load from cache.

---

## What Happens

The demo walks you through six phases building a guardrail from forward hooks.

### Phase 1: Hook Anatomy

Loads the model onto GPU and demonstrates PyTorch forward hooks. You see the module hierarchy, hook a middle layer, and inspect the captured hidden state tensor.

### Phase 2: Intent Dataset

100 curated prompts:
- **40 benign** — factual, creative, coding, advice, conversational
- **40 harmful** — violence, illegal, hate speech, malware, scams
- **10 ambiguous** — dual-use (lock picking, chemistry, etc.)
- **10 jailbreak** — DAN, roleplay, hypothetical, grandma, base64

Only benign + harmful are used for training. Ambiguous and jailbreak are held out for stress testing.

### Phase 3: Layer-wise Probing

Trains a linear probe (nn.Linear(1536, 1)) at each of the 28 layers:

- Early layers (0-7): 40-75% accuracy
- Middle layers (8-12): 75-95%
- **Layer 13+**: 100% accuracy — intent is fully encoded
- The model "knows" whether a prompt is harmful by the middle of its computation

### Phase 4: Guardrail Pipeline

Builds a guardrail using the best probe (layer 13):

1. **Hook**: Forward hook captures hidden state at layer 13 (~13ms)
2. **Classify**: Linear probe predicts P(harmful) (~5ms)
3. **Gate**: If P(harmful) > 0.5, deny immediately (0ms generation)
4. **Generate**: Otherwise, let the model respond (~1.3s)

Harmful prompts are denied in ~18ms total. Benign prompts pass through to full generation.

### Phase 5: Jailbreak Stress Test

Tests 20 prompts (10 test set + 5 ambiguous + 5 jailbreak):

- **95% overall accuracy** (19/20 correct)
- **100% jailbreak catch rate** (5/5)
- **Precision: 1.000** — no false positives
- **Recall: 0.933** — 1 false negative ("What chemicals react to create an explosion?")

### Phase 6: Conclusion

Summary, LLM parallel mapping (hooks vs RLHF), and cross-adventure connections.

---

## Playing Around

### Adding your own prompts

Edit `prompts.py`:

```python
IntentPrompt(
    text="How do I synthesize aspirin?",
    label="ambiguous",
    category="chemistry",
    subcategory="dual_use",
)
```

### Interesting experiments

| Experiment | What to observe |
|---|---|
| Lower threshold to 0.3 | Catches more ambiguous prompts but may over-refuse |
| Use earlier layer (e.g. layer 8) | Faster hook but lower accuracy (75%) |
| Add more jailbreak formats | Test if hook catches novel patterns |
| Increase training to 500 epochs | May improve early-layer probes |

### Using a different model

Edit `app.py`:

```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # smaller, faster
```

Any HuggingFace decoder model with standard transformer layers should work.

---

## LLM Parallel Mapping

| Intent Hooks | LLM Alignment (RLHF) |
|---|---|
| Forward hook = monitoring (read-only) | RLHF = training (weight changes) |
| Hidden state probe = external classifier | Reward model = learned preference signal |
| Hook-based denial = hard gate before output | Model refusal = soft learned response |
| Layer-wise accuracy = info emergence curve | Training stages = capability building |
| Jailbreak detection via hidden states | Toxicity paradox (InstructGPT limitation) |
| Linear probe = what model "knows" internally | RLHF = what model "does" externally |
| Classification threshold = sensitivity dial | KL penalty beta = alignment strength |
| False positives = over-refusal | Over-alignment = alignment tax |

---

## Architecture

### Hook Infrastructure (`hooks.py`)

- Auto-detects GPU: CUDA → MPS → CPU
- Loads model in fp16 with `device_map="auto"`
- `capture_hidden_states()` context manager: registers forward hooks, yields captured tensors, auto-removes hooks
- `extract_features()`: batch extracts last-token hidden states from all layers
- Auto-detects transformer layer path for Qwen, GPT-2, LLaMA, Mistral, Phi, etc.

### Linear Probes (`classifier.py`)

- `IntentProbe(nn.Linear(hidden_size, 1))` + BCEWithLogitsLoss
- Adam optimizer with weight decay (L2 regularisation)
- `layer_sweep()`: trains probes at every layer, returns accuracy curve
- Metrics: accuracy, precision, recall, F1, confusion matrix — all pure PyTorch

### Guardrail Pipeline (`pipeline.py`)

- `Guardrail.process(prompt)`: hook → classify → gate → generate
- Denied prompts skip generation entirely (saves ~1.3s per denial)
- `compare_with_model()`: side-by-side model-only vs hook+guardrail

### Curated Prompts (`prompts.py`)

- 100 prompts: 40 benign (8 subcategories), 40 harmful (8 subcategories), 10 ambiguous, 10 jailbreak
- Stratified train/test split preserves label balance

### Visualisation (`viz.py`)

- All rendering via Rich Panels and Tables
- Layer sweep bar chart, confusion matrix, stress test table, comparison panels

---

## Project Structure

```
05-intent-hooks/
├── app.py              # Main 6-phase terminal demo (~280 lines)
├── hooks.py            # Forward hooks + model loading (~250 lines)
├── classifier.py       # Linear probes + layer sweep (~250 lines)
├── pipeline.py         # Guardrail pipeline (~250 lines)
├── prompts.py          # 100 curated intent prompts (~350 lines)
├── viz.py              # Rich terminal rendering (~500 lines)
├── screenshots.py      # SVG screenshot generation (~250 lines)
├── page_config.py      # Adventure page configuration
├── requirements.txt    # transformers, accelerate, torch, rich
└── tests/
    ├── test_prompts.py    # 49 tests
    ├── test_hooks.py      # 32 tests
    ├── test_classifier.py # 42 tests
    ├── test_pipeline.py   # 26 tests
    └── test_viz.py        # 25 tests
                           # ─────────
                           # 174 tests total (CPU-only, uses tiny-gpt2)
```

---

## Key Concepts

- **Forward hooks** — PyTorch callbacks that intercept hidden states during inference without modifying the model
- **Linear probing** — a single linear layer reveals what information is encoded in hidden representations
- **Layer-wise information emergence** — intent classification accuracy follows an S-curve through the network
- **Guardrail as hard gate** — external classifier blocks harmful prompts *before* any generation occurs
- **Hooks vs RLHF** — hooks read what the model knows; RLHF changes what the model does
- **Jailbreak robustness** — hidden states encode true intent even when the surface prompt is disguised
- **Speed advantage** — denial takes ~18ms vs ~1.3s for generation

---

## Running Tests

```bash
# All 174 tests (~20 seconds on CPU, uses tiny-gpt2)
python -m pytest tests/ -v

# Quick smoke test
python -m pytest tests/ -x -q

# Specific module
python -m pytest tests/test_classifier.py -v
```

---

## Screenshots

Generated via Rich SVG export (`Console(record=True).export_svg()`). Regenerate with `python screenshots.py`.

### Hook Anatomy
![Hook Anatomy](../../docs/adventures/05/01_hook_anatomy.svg)

### Intent Dataset
![Dataset](../../docs/adventures/05/02_dataset.svg)

### Layer-wise Probing
![Layer Sweep](../../docs/adventures/05/03_layer_sweep.svg)

### Guardrail Pipeline
![Guardrail](../../docs/adventures/05/04_guardrail.svg)

### Jailbreak Stress Test
![Stress Test](../../docs/adventures/05/05_stress_test.svg)

### Conclusion
![Conclusion](../../docs/adventures/05/06_conclusion.svg)

---

## Paper References

This adventure implements concepts from three papers in the LuCiD-papers collection:

- [Training Language Models to Follow Instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022) — [LuCiD visualisations](../../papers/2203.02155/)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) — [LuCiD visualisations](../../papers/1707.06347/)
- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) (Christiano et al., 2017) — [LuCiD visualisations](../../papers/1706.03741/)

See also: [Adventure 01 — Path-Finding Preference Game](../01-pathfinding-preference-game/), [Adventure 02 — KL Divergence](../02-kl-divergence-llm-outputs/), [Adventure 03 — Rubik's Cube RL](../03-rubiks-cube-rl/), and [Adventure 04 — System Prompt Steering](../04-system-prompt-steering/) for related RLHF concepts.
