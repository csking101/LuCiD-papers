# 2203.02155 -- Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

**Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)

**Paper:** [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

**Notes:** Obsidian vault (private)

## Key Idea

InstructGPT applies the RLHF pipeline from the summarization paper (Stiennon et al. 2020) to the much broader problem of following arbitrary human instructions. The same three-step recipe -- SFT on labeler demonstrations, reward model from K-way rankings, PPO fine-tuning -- but with two key innovations: PPO-ptx (mixing pretraining gradients to prevent alignment tax) and batched K-way ranking (grouping C(K,2) pairs per prompt as a single gradient step to prevent overfitting). The headline result: a 1.3B InstructGPT model is preferred by humans over the 175B GPT-3 baseline. InstructGPT is the direct ancestor of ChatGPT.

## Visualizations

### Manim Animations (MP4)

| # | Script | Description | Output |
|---|--------|-------------|--------|
| 1 | `01_instructgpt_pipeline.py` | Full InstructGPT pipeline: SFT → RM (K-way) → PPO-ptx | [MP4](../../docs/papers/2203.02155/InstructGPTPipeline.mp4) |
| 2 | `02_ppo_ptx_objective.py` | PPO-ptx objective derivation: reward + KL + pretraining | [MP4](../../docs/papers/2203.02155/PPOptxObjective.mp4) |
| 3 | `03_kway_ranking_batching.py` | K-way ranking & batching strategy: naive vs batched | [MP4](../../docs/papers/2203.02155/KwayRankingBatching.mp4) |
| 10 | `10_alignment_timeline.py` | Alignment timeline: RLHF → PPO → Summarize HF → InstructGPT → ChatGPT → DPO | [MP4](../../docs/papers/2203.02155/AlignmentTimeline.mp4) |

### Static Figures (PNG)

| # | Script | Description | Output |
|---|--------|-------------|--------|
| 4 | `04_task_distribution_dataset.py` | API prompt categories (donut) + dataset sizes (stacked bar) | [PNG](output/static/04_task_distribution_dataset.png) |
| 5 | `05_winrate_vs_baselines.py` | Winrate against 175B SFT for all model variants | [PNG](output/static/05_winrate_vs_baselines.png) |
| 6 | `06_truthfulqa_results.py` | TruthfulQA: truthful & truthful+informative fractions | [PNG](output/static/06_truthfulqa_results.png) |
| 7 | `07_toxicity_paradox.py` | Toxicity paradox across 3 prompt conditions | [PNG](output/static/07_toxicity_paradox.png) |
| 8 | `08_alignment_tax_ppoptx.py` | Alignment tax: PPO regresses, PPO-ptx recovers | [PNG](output/static/08_alignment_tax_ppoptx.png) |
| 9 | `09_who_are_we_aligning_to.py` | Four-layer alignment hierarchy infographic | [PNG](output/static/09_who_are_we_aligning_to.png) |

### Interactive Demos (HTML)

| # | Script | Description | Output | Live Demo |
|---|--------|-------------|--------|-----------|
| 4 | `04_task_distribution_dataset.py` | Task distribution + dataset composition | [HTML](output/interactive/04_task_distribution_dataset.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/2203.02155/#viz-04) |
| 5 | `05_winrate_vs_baselines.py` | Winrate bar chart with annotations | [HTML](output/interactive/05_winrate_vs_baselines.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/2203.02155/#viz-05) |
| 6 | `06_truthfulqa_results.py` | TruthfulQA with 175B/all-sizes toggle | [HTML](output/interactive/06_truthfulqa_results.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/2203.02155/#viz-06) |
| 7 | `07_toxicity_paradox.py` | Toxicity paradox with shaded regions | [HTML](output/interactive/07_toxicity_paradox.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/2203.02155/#viz-07) |
| 8 | `08_alignment_tax_ppoptx.py` | NLP benchmarks: GPT vs SFT vs PPO vs PPO-ptx | [HTML](output/interactive/08_alignment_tax_ppoptx.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/2203.02155/#viz-08) |

## Running

```bash
# From this directory (papers/2203.02155/scripts/)
cd scripts

# Static + interactive (use venv python for correct dependencies)
../../.venv/bin/python 04_task_distribution_dataset.py
../../.venv/bin/python 05_winrate_vs_baselines.py
../../.venv/bin/python 06_truthfulqa_results.py
../../.venv/bin/python 07_toxicity_paradox.py
../../.venv/bin/python 08_alignment_tax_ppoptx.py
../../.venv/bin/python 09_who_are_we_aligning_to.py

# Animations (use venv manim)
../../.venv/bin/manim -qm --media_dir ../output/animations 01_instructgpt_pipeline.py InstructGPTPipeline
../../.venv/bin/manim -qm --media_dir ../output/animations 02_ppo_ptx_objective.py PPOptxObjective
../../.venv/bin/manim -qm --media_dir ../output/animations 03_kway_ranking_batching.py KwayRankingBatching
../../.venv/bin/manim -qm --media_dir ../output/animations 10_alignment_timeline.py AlignmentTimeline
```
