# 2009.01325 -- Learning to Summarize from Human Feedback

**Authors:** Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul F. Christiano (2020)

**Paper:** [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)

**Notes:** Obsidian vault (private)

## Key Idea

This paper demonstrates that optimizing for human preferences via reinforcement learning produces significantly better summaries than supervised fine-tuning alone. The pipeline: (1) SFT a GPT-3-style model on Reddit TL;DR, (2) train a reward model from 65K human comparisons using a Bradley-Terry loss, (3) fine-tune with PPO using the learned reward with a KL penalty against the SFT policy. The RL-trained models generate summaries preferred by humans over the reference TL;DRs and even over much larger supervised models. The paper also reveals reward overoptimization — past a certain KL budget, the RM score keeps climbing while true quality degrades.

## Visualizations

### Manim Animations (MP4)

| # | Script | Description | Output |
|---|--------|-------------|--------|
| 1 | `01_rlhf_text_pipeline.py` | Full RLHF-for-text pipeline: SFT → RM → PPO | [MP4](../../docs/papers/2009.01325/RLHFTextPipeline.mp4) |
| 2 | `02_bradley_terry_rm.py` | Bradley-Terry reward model loss derivation | [MP4](../../docs/papers/2009.01325/BradleyTerryRM.mp4) |
| 3 | `03_kl_penalized_reward.py` | KL-penalized reward objective visualization | [MP4](../../docs/papers/2009.01325/KLPenalizedReward.mp4) |
| 10 | `10_alignment_timeline.py` | Alignment research timeline: RLHF → PPO → this paper → InstructGPT → DPO | [MP4](../../docs/papers/2009.01325/AlignmentTimeline.mp4) |

### Static Figures (PNG)

| # | Script | Description | Output |
|---|--------|-------------|--------|
| 4 | `04_sft_vs_rl_quality.py` | SFT vs RL summary quality comparison | [PNG](output/static/04_sft_vs_rl_quality.png) |
| 5 | `05_rm_accuracy_scaling.py` | Reward model accuracy scaling with data size | [PNG](output/static/05_rm_accuracy_scaling.png) |
| 6 | `06_reward_overoptimization.py` | Reward overoptimization: RM score vs true quality | [PNG](output/static/06_reward_overoptimization.png) |
| 7 | `07_kl_coefficient_effect.py` | Effect of KL penalty coefficient on quality | [PNG](output/static/07_kl_coefficient_effect.png) |
| 8 | `08_rouge_vs_learned_rm.py` | ROUGE vs learned RM as optimization target | [PNG](output/static/08_rouge_vs_learned_rm.png) |
| 9 | `09_transfer_cnn_dm.py` | Transfer learning: TL;DR model on CNN/DM | [PNG](output/static/09_transfer_cnn_dm.png) |

### Interactive Demos (HTML)

| # | Script | Description | Output | Live Demo |
|---|--------|-------------|--------|-----------|
| 4 | `04_sft_vs_rl_quality.py` | SFT vs RL quality explorer | [HTML](output/interactive/04_sft_vs_rl_quality.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/2009.01325/#viz-04) |
| 5 | `05_rm_accuracy_scaling.py` | RM accuracy scaling explorer | [HTML](output/interactive/05_rm_accuracy_scaling.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/2009.01325/#viz-05) |
| 6 | `06_reward_overoptimization.py` | Overoptimization dynamics explorer | [HTML](output/interactive/06_reward_overoptimization.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/2009.01325/#viz-06) |
| 7 | `07_kl_coefficient_effect.py` | KL coefficient slider | [HTML](output/interactive/07_kl_coefficient_effect.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/2009.01325/#viz-07) |
| 8 | `08_rouge_vs_learned_rm.py` | ROUGE vs learned RM comparison | [HTML](output/interactive/08_rouge_vs_learned_rm.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/2009.01325/#viz-08) |

## Running

```bash
# From this directory (papers/2009.01325/scripts/)
cd scripts

# Static + interactive (use venv python for correct dependencies)
../../.venv/bin/python 04_sft_vs_rl_quality.py
../../.venv/bin/python 05_rm_accuracy_scaling.py
../../.venv/bin/python 06_reward_overoptimization.py
../../.venv/bin/python 07_kl_coefficient_effect.py
../../.venv/bin/python 08_rouge_vs_learned_rm.py
../../.venv/bin/python 09_transfer_cnn_dm.py

# Animations (use venv manim)
../../.venv/bin/manim -qm --media_dir ../output/animations 01_rlhf_text_pipeline.py RLHFTextPipeline
../../.venv/bin/manim -qm --media_dir ../output/animations 02_bradley_terry_rm.py BradleyTerryRM
../../.venv/bin/manim -qm --media_dir ../output/animations 03_kl_penalized_reward.py KLPenalizedReward
../../.venv/bin/manim -qm --media_dir ../output/animations 10_alignment_timeline.py AlignmentTimeline
```
