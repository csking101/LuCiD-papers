# 1706.03741 -- Deep Reinforcement Learning from Human Preferences

**Authors:** Paul Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, Dario Amodei (2017)

**Paper:** [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)

**Notes:** Obsidian vault (private)

## Key Idea

Train an RL agent without a hand-designed reward function. Instead, a human watches pairs of short trajectory clips and indicates which one is preferred. A reward model is trained on these preferences using the Bradley-Terry model + cross-entropy loss, and the policy optimizes this learned reward.

## Visualizations

### Manim Animations (MP4)

| # | Script | Description | Output |
|---|--------|-------------|--------|
| 1 | `01_system_architecture.py` | Three asynchronous processes (Figure 1) | [MP4](../../docs/papers/1706.03741/SystemArchitecture.mp4) |
| 2 | `02_rl_vs_rlhf_pipeline.py` | Traditional RL vs RLHF side-by-side | [MP4](../../docs/papers/1706.03741/RLvsRLHF.mp4) |
| 3 | `03_preference_elicitation.py` | 5-step preference collection flow | [MP4](../../docs/papers/1706.03741/PreferenceElicitation.mp4) |
| 6 | `06_reward_convergence.py` | Reward model improving with more data | [MP4](../../docs/papers/1706.03741/RewardConvergence.mp4) |

### Static Figures (PNG)

| # | Script | Description | Output |
|---|--------|-------------|--------|
| 4 | `04_bradley_terry.py` | Bradley-Terry sigmoid + human error | [PNG](output/static/04_bradley_terry.png) |
| 5 | `05_cross_entropy_loss.py` | Loss curves + 3D loss surface | [PNG](output/static/05_cross_entropy_loss.png) |
| 7 | `07_mujoco_results.py` | 8-panel MuJoCo results (Figure 2) | [PNG](output/static/07_mujoco_results.png) |
| 8 | `08_atari_results.py` | 7-panel Atari results (Figure 3) | [PNG](output/static/08_atari_results.png) |
| 9 | `09_ablation_heatmap.py` | Ablation study heatmap | [PNG](output/static/09_ablation_heatmap.png) |

### Interactive Demos (HTML)

| # | Script | Description | Output | Live Demo |
|---|--------|-------------|--------|-----------|
| 4 | `04_bradley_terry.py` | Epsilon slider, hover values | [HTML](output/interactive/04_bradley_terry.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/1706.03741/#viz-04) |
| 5 | `05_cross_entropy_loss.py` | 3D rotatable loss surface | [HTML](output/interactive/05_cross_entropy_loss.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/1706.03741/#viz-05) |
| 9 | `09_ablation_heatmap.py` | Hover-detail ablation heatmap | [HTML](output/interactive/09_ablation_heatmap.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/1706.03741/#viz-09) |
| 10 | `10_preference_demo.py` | Full RLHF preference simulation | [HTML](output/interactive/10_preference_demo.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/1706.03741/#viz-10) |

## Running

```bash
# From this directory (papers/1706.03741/scripts/)
cd scripts

# Static + interactive
python 04_bradley_terry.py
python 05_cross_entropy_loss.py
python 07_mujoco_results.py
python 08_atari_results.py
python 09_ablation_heatmap.py
python 10_preference_demo.py

# Animations
manim -ql --media_dir ../output/animations 01_system_architecture.py SystemArchitecture
manim -ql --media_dir ../output/animations 02_rl_vs_rlhf_pipeline.py RLvsRLHF
manim -ql --media_dir ../output/animations 03_preference_elicitation.py PreferenceElicitation
manim -ql --media_dir ../output/animations 06_reward_convergence.py RewardConvergence
```
