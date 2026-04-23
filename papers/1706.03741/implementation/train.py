# Training Loop
#
# The main RLHF training loop that orchestrates the 3 processes:
#
#   Process 1: Policy interacts with environment, collects trajectories
#              Policy parameters updated to maximize predicted reward r_hat
#
#   Process 2: Segments are selected from trajectories, paired, and
#              sent for comparison (synthetic oracle or human)
#
#   Process 3: Reward model r_hat is updated via supervised learning
#              on the preference database
#
# In the paper these run asynchronously. Here we run them sequentially
# per iteration for simplicity:
#
#   for each iteration:
#       1. Collect trajectories using current policy
#       2. Slice segments, pair them, get preferences
#       3. Update reward model on preference database
#       4. Update policy using new reward model
#       5. Log metrics
