# Policy
#
# A policy gradient agent that optimizes against the learned reward
# model r_hat. Since the reward function is non-stationary (it changes
# as more preferences are collected), policy gradient methods are
# preferred over value-based methods.
#
# Options (pick one to implement):
#   - REINFORCE (simplest — good starting point)
#   - Actor-Critic (closer to the paper's A2C)
#
# The paper uses:
#   - A2C for Atari
#   - TRPO for MuJoCo
# For this grid world, REINFORCE or a simple actor-critic should suffice.
#
# Remember: normalize rewards from r_hat to zero mean, constant std
# before feeding them to the policy (Section 2.1 of the paper).
