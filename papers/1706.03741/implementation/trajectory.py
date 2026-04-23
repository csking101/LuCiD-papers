# Trajectory Collection
#
# Roll out the current policy to collect trajectories.
# Slice trajectories into fixed-length segments.
# Pair segments for human comparison.
#
# Key data structures:
#   trajectory = list of (observation, action) tuples
#   segment = trajectory[i:i+k] for some fixed k (SEGMENT_LENGTH)
#   pair = (segment_1, segment_2)
