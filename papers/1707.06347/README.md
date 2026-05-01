# 1707.06347 -- Proximal Policy Optimization Algorithms

**Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)

**Paper:** [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

**Notes:** Obsidian vault (private)

## Key Idea

PPO proposes a clipped surrogate objective that constrains policy updates, achieving the stability of trust-region methods (TRPO) with the simplicity of first-order optimization. Instead of solving a constrained optimization with second-order methods, PPO clips the importance sampling ratio to [1-epsilon, 1+epsilon], preventing destructively large updates. Combined with GAE and multiple epochs of minibatch updates, PPO became the default RL optimizer for LLM alignment via RLHF.

## Visualizations

### Manim Animations (MP4)

| # | Script | Description | Output |
|---|--------|-------------|--------|
| 1 | `01_stats_refresher.py` | Statistics refresher: expectation, variance, log-derivative trick | [MP4](../../docs/papers/1707.06347/StatsRefresher.mp4) |
| 2 | `02_policy_gradient_intuition.py` | REINFORCE algorithm intuition | [MP4](../../docs/papers/1707.06347/PolicyGradientIntuition.mp4) |
| 3 | `03_policy_gradient_derivation.py` | Step-by-step PG derivation | [MP4](../../docs/papers/1707.06347/PolicyGradientDerivation.mp4) |
| 6 | `06_trust_region_motivation.py` | Parameter-space vs policy-space distances | [MP4](../../docs/papers/1707.06347/TrustRegionMotivation.mp4) |
| 9 | `09_ppo_vs_trpo_vs_pg.py` | PPO vs TRPO vs vanilla PG comparison | [MP4](../../docs/papers/1707.06347/PPOvsTRPOvsPG.mp4) |

### Static Figures (PNG)

| # | Script | Description | Output |
|---|--------|-------------|--------|
| 4 | `04_advantage_vs_raw_reward.py` | Advantage vs raw reward variance reduction | [PNG](output/static/04_advantage_vs_raw_reward.png) |
| 5 | `05_gae_lambda_tradeoff.py` | GAE lambda bias-variance tradeoff | [PNG](output/static/05_gae_lambda_tradeoff.png) |
| 7 | `07_kl_trust_region.py` | KL divergence trust region visualization | [PNG](output/static/07_kl_trust_region.png) |
| 8 | `08_clipped_surrogate.py` | Clipped surrogate objective mechanism | [PNG](output/static/08_clipped_surrogate.png) |
| 10 | `10_ppo_llm_alignment.py` | PPO in the RLHF pipeline for LLMs | [PNG](output/static/10_ppo_llm_alignment.png) |

### Interactive Demos (HTML)

| # | Script | Description | Output | Live Demo |
|---|--------|-------------|--------|-----------|
| 4 | `04_advantage_vs_raw_reward.py` | Variance comparison explorer | [HTML](output/interactive/04_advantage_vs_raw_reward.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/1707.06347/#viz-04) |
| 5 | `05_gae_lambda_tradeoff.py` | Lambda slider, weight decay curves | [HTML](output/interactive/05_gae_lambda_tradeoff.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/1707.06347/#viz-05) |
| 7 | `07_kl_trust_region.py` | Trust region radius explorer | [HTML](output/interactive/07_kl_trust_region.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/1707.06347/#viz-07) |
| 8 | `08_clipped_surrogate.py` | Epsilon slider, A>0 and A<0 cases | [HTML](output/interactive/08_clipped_surrogate.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/1707.06347/#viz-08) |
| 10 | `10_ppo_llm_alignment.py` | RLHF pipeline stages explorer | [HTML](output/interactive/10_ppo_llm_alignment.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/1707.06347/#viz-10) |
| 11 | `11_ppo_cartpole_demo.py` | Self-contained PPO CartPole in browser | [HTML](output/interactive/11_ppo_cartpole_demo.html) | [Open](https://csking101.github.io/LuCiD-papers/papers/1707.06347/#viz-11) |

## Running

```bash
# From this directory (papers/1707.06347/scripts/)
cd scripts

# Static + interactive (use venv python for correct dependencies)
../../.venv/bin/python 04_advantage_vs_raw_reward.py
../../.venv/bin/python 05_gae_lambda_tradeoff.py
../../.venv/bin/python 07_kl_trust_region.py
../../.venv/bin/python 08_clipped_surrogate.py
../../.venv/bin/python 10_ppo_llm_alignment.py
../../.venv/bin/python 11_ppo_cartpole_demo.py

# Animations (use venv manim)
../../.venv/bin/manim -qm --media_dir ../output/animations 01_stats_refresher.py StatsRefresher
../../.venv/bin/manim -qm --media_dir ../output/animations 02_policy_gradient_intuition.py PolicyGradientIntuition
../../.venv/bin/manim -qm --media_dir ../output/animations 03_policy_gradient_derivation.py PolicyGradientDerivation
../../.venv/bin/manim -qm --media_dir ../output/animations 06_trust_region_motivation.py TrustRegionMotivation
../../.venv/bin/manim -qm --media_dir ../output/animations 09_ppo_vs_trpo_vs_pg.py PPOvsTRPOvsPG
```
