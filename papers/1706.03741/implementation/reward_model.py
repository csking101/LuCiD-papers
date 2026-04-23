# Reward Model
#
# An ensemble of neural networks that predict scalar rewards from
# (observation, action) pairs.
#
# Trained using the Bradley-Terry model + cross-entropy loss:
#
#   P[sigma_1 > sigma_2] = exp(sum r(o,a) for sigma_1)
#                          / (exp(sum for sigma_1) + exp(sum for sigma_2))
#
#   loss = -sum[ mu(1)*log(P[s1>s2]) + mu(2)*log(P[s2>s1]) ]
#
# The ensemble provides uncertainty estimates for active query selection.
# Each predictor is trained on a bootstrap sample of the preference DB.
# Final reward = average of independently normalized predictor outputs.
