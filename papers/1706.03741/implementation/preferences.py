# Preference Collection
#
# Manages the preference database D of triples (sigma_1, sigma_2, mu).
# Provides a synthetic oracle for automated testing.
#
# The synthetic oracle compares segments using a hidden "true" reward
# (e.g., negative distance to goal), so you can verify the algorithm
# works before hooking up real human feedback.
#
# mu is the preference distribution over {1, 2}:
#   - Human prefers sigma_1: mu = [1, 0]
#   - Human prefers sigma_2: mu = [0, 1]
#   - Equal preference:      mu = [0.5, 0.5]
#   - Incomparable:          not added to database
