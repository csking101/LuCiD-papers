# Configuration / Hyperparameters
# Adjust these as you experiment.

# --- Environment ---
GRID_HEIGHT = 8
GRID_WIDTH = 8
TERMINAL_OBJECTS_PLACEMENT = {(GRID_WIDTH-1,GRID_HEIGHT-1):100,(1,0):-5}
WALL_PLACEMENT = {(1,1)}
MAX_STEPS_PER_EPISODE = 50
STEP_PENALTY = -1.0          # true reward for non-terminal steps (used by oracle + DP)

# --- Trajectories ---
SEGMENT_LENGTH = 10          # target segment length (like the 1-2 sec clips in the paper)
MIN_SEGMENT_LENGTH = 3       # minimum viable segment (allows short-episode segments)
TRAJECTORIES_PER_ITER = 20   # how many episodes to collect per training iteration

# --- Preferences ---
PAIRS_PER_ITER = 20          # how many segment pairs to query per iteration
PREFERENCE_DB_MAX = 5000     # max size of the preference database

# --- Reward Model ---
REWARD_ENSEMBLE_SIZE = 3     # number of reward predictors in the ensemble
REWARD_HIDDEN_SIZE = 64      # hidden layer size for reward network
REWARD_LR = 1e-3             # learning rate for reward model
REWARD_EPOCHS = 3            # training epochs per reward model update (was 10, reduced for speed)
REWARD_BATCH_SIZE = 32
HUMAN_ERROR_RATE = 0.1       # epsilon: probability of random human response (Section 2.2)
DEVICE = "cpu"               # force CPU — CUDA kernel-launch overhead kills tiny tensor ops

# --- Policy ---
POLICY_HIDDEN_SIZE = 64
POLICY_LR = 5e-4             # moderate LR — low enough for stability, high enough to learn
POLICY_EPOCHS = 2            # 2 epochs per update (compromise: signal strength vs stability)
GAMMA = 0.99                 # discount factor
ENTROPY_BETA_START = 0.1     # initial entropy bonus (high for exploration)
ENTROPY_BETA_END = 0.005     # final entropy bonus (low for exploitation)
EXPLORATION_EPSILON = 0.25   # random action rate for EXPLORATION trajectories (preferences)
GRAD_CLIP_NORM = 1.0         # max gradient norm for policy updates
POLICY_UPDATE_INTERVAL = 2   # update policy every N iterations

# --- Warm-up ---
WARMUP_TRAJECTORIES = 100    # random trajectories to seed preference DB before policy updates

# --- Observability ---
EVAL_INTERVAL = 5            # full DP comparison every N iterations

# --- Human Feedback ---
HUMAN_FEEDBACK_INTERVAL = 25 # iterations between human feedback rounds (0 = disabled)
HUMAN_PAIRS_PER_ROUND = 5   # segment pairs shown to human per round

# --- Training ---
NUM_ITERATIONS = 100         # total RLHF training iterations
