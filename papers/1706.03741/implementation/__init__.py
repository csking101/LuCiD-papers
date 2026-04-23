# Cat Grid World — RLHF Implementation
#
# Full implementation of the RLHF algorithm from 1706.03741
# applied to a simple grid environment with a cat agent.
#
# Files:
#   env.py           - Grid environment (Gymnasium-style API)
#   trajectory.py    - Rollouts, segment slicing, pairing
#   preferences.py   - Preference DB + synthetic oracle
#   reward_model.py  - Ensemble reward networks, Bradley-Terry loss
#   policy.py        - Policy gradient agent
#   train.py         - Main RLHF training loop (3 processes)
#   export.py        - Export results to JSON for JS demo
#   config.py        - Hyperparameters
