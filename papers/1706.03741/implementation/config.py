# Configuration / Hyperparameters
# Adjust these as you experiment.

# --- Environment ---
GRID_SIZE = 8
MAX_STEPS_PER_EPISODE = 50

# --- Trajectories ---
SEGMENT_LENGTH = 10          # number of steps per segment (like the 1-2 sec clips in the paper)
TRAJECTORIES_PER_ITER = 20   # how many episodes to collect per training iteration

# --- Preferences ---
PAIRS_PER_ITER = 10          # how many segment pairs to query per iteration
PREFERENCE_DB_MAX = 5000     # max size of the preference database

# --- Reward Model ---
REWARD_ENSEMBLE_SIZE = 3     # number of reward predictors in the ensemble
REWARD_HIDDEN_SIZE = 64      # hidden layer size for reward network
REWARD_LR = 1e-3             # learning rate for reward model
REWARD_EPOCHS = 10           # training epochs per reward model update
REWARD_BATCH_SIZE = 32
HUMAN_ERROR_RATE = 0.1       # epsilon: probability of random human response (Section 2.2)

# --- Policy ---
POLICY_HIDDEN_SIZE = 64
POLICY_LR = 1e-3
POLICY_EPOCHS = 5            # policy update epochs per iteration
GAMMA = 0.99                 # discount factor

# --- Training ---
NUM_ITERATIONS = 100         # total RLHF training iterations
