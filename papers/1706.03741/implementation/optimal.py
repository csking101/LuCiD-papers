# Optimal Baseline via Dynamic Programming
#
# Computes the exact optimal policy and value function for the grid world
# using value iteration. Since the environment is deterministic (action a
# in state s always leads to the same s'), this converges quickly.
#
# The step reward function matches the SyntheticOracle:
#   - terminal_value if the step lands on a terminal cell (episode ends)
#   - step_penalty + proximity_bonus for non-terminal steps (distance shaping)
#
# This gives us ground-truth targets to compare the learned policy against:
#   - Policy accuracy:  % of cells where learned best-action == optimal best-action
#   - Reward MSE:       mean squared error between learned r_hat(s,a) and Q*(s,a)
#   - True return:      average discounted return of the learned policy under true reward
#   - Optimal return:   V*(start) — the best achievable return

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import Action, DELTAS


def _get_next_state(x, y, action_int, env):
    """
    Compute the next state (nx, ny) given current (x, y) and action.
    Respects boundaries and walls. Returns (nx, ny).
    """
    action = Action(action_int)
    dx, dy = DELTAS[action]
    nx, ny = x + dx, y + dy

    # Boundary check
    if nx < 0 or nx >= env.width or ny < 0 or ny >= env.height:
        return x, y  # stay in place

    # Wall check
    if (nx, ny) in env.wall_placement:
        return x, y  # stay in place

    return nx, ny


def _step_reward(nx, ny, terminals, step_penalty, goal_pos, max_dist,
                 use_distance_shaping):
    """
    Compute the true reward for transitioning to (nx, ny).
    Matches the SyntheticOracle's true_step_reward.
    """
    if (nx, ny) in terminals:
        return float(terminals[(nx, ny)])

    if use_distance_shaping and goal_pos is not None:
        dist = abs(nx - goal_pos[0]) + abs(ny - goal_pos[1])
        proximity_bonus = (max_dist - dist) / max_dist
        return step_penalty + proximity_bonus

    return step_penalty


def value_iteration(env, terminals: dict, step_penalty: float = -1.0,
                    gamma: float = 0.99, use_distance_shaping: bool = True,
                    tol: float = 1e-8, max_iters: int = 1000) -> dict:
    """
    Compute V*(s) for every non-wall cell using value iteration.

    Terminal cells are absorbing: V(terminal) = 0 (the terminal reward is
    received on the transition INTO the terminal, not from it).

    Returns: dict mapping (x, y) -> V* value.
    """
    # Find goal position for distance shaping
    goal_pos = None
    max_dist = 1
    if terminals and use_distance_shaping:
        goal_pos = max(terminals, key=terminals.get)
        max_dist = abs(goal_pos[0]) + abs(goal_pos[1]) + 2

    # Initialize V(s) = 0 for all states
    V = {}
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) not in env.wall_placement:
                V[(x, y)] = 0.0

    for iteration in range(max_iters):
        delta = 0.0
        for (x, y) in list(V.keys()):
            # Terminal cells are absorbing — no further reward
            if (x, y) in terminals:
                continue

            best_value = float("-inf")
            for a in range(4):
                nx, ny = _get_next_state(x, y, a, env)

                r = _step_reward(nx, ny, terminals, step_penalty,
                                 goal_pos, max_dist, use_distance_shaping)

                if (nx, ny) in terminals:
                    # Terminal is absorbing: no future value after reaching it
                    q = r
                else:
                    q = r + gamma * V[(nx, ny)]

                best_value = max(best_value, q)

            delta = max(delta, abs(best_value - V[(x, y)]))
            V[(x, y)] = best_value

        if delta < tol:
            break

    return V


def compute_q_values(env, V_star: dict, terminals: dict,
                     step_penalty: float = -1.0, gamma: float = 0.99,
                     use_distance_shaping: bool = True) -> dict:
    """
    Compute Q*(s, a) for every non-wall cell and action.

    Returns: dict mapping (x, y, a) -> Q-value.
    """
    goal_pos = None
    max_dist = 1
    if terminals and use_distance_shaping:
        goal_pos = max(terminals, key=terminals.get)
        max_dist = abs(goal_pos[0]) + abs(goal_pos[1]) + 2

    Q = {}
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) in env.wall_placement:
                continue
            if (x, y) in terminals:
                # From a terminal cell, no actions yield future reward
                for a in range(4):
                    Q[(x, y, a)] = 0.0
                continue

            for a in range(4):
                nx, ny = _get_next_state(x, y, a, env)
                r = _step_reward(nx, ny, terminals, step_penalty,
                                 goal_pos, max_dist, use_distance_shaping)

                if (nx, ny) in terminals:
                    Q[(x, y, a)] = r
                else:
                    Q[(x, y, a)] = r + gamma * V_star[(nx, ny)]

    return Q


def get_optimal_policy(Q_star: dict, env) -> dict:
    """
    Extract the optimal deterministic policy from Q*.

    Returns: dict mapping (x, y) -> best action int.
    """
    pi = {}
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) in env.wall_placement:
                continue
            best_a = 0
            best_q = float("-inf")
            for a in range(4):
                q = Q_star.get((x, y, a), float("-inf"))
                if q > best_q:
                    best_q = q
                    best_a = a
            pi[(x, y)] = best_a
    return pi


def compare_policy_accuracy(policy, pi_star: dict, env) -> float:
    """
    Fraction of non-wall, non-terminal cells where the learned policy's
    best action matches the optimal policy's best action.
    """
    match = 0
    total = 0
    terminals = env.terminal_objects_placement

    for (x, y), optimal_action in pi_star.items():
        if (x, y) in terminals:
            continue  # don't count terminal cells
        probs = policy.get_action_probabilities([x, y])
        learned_best = probs.index(max(probs))
        if learned_best == optimal_action:
            match += 1
        total += 1

    return match / max(total, 1)


def compare_reward_mse(reward_ensemble, Q_star: dict, env) -> float:
    """
    Mean squared error between the learned reward r_hat(s,a) and Q*(s,a)
    across all (state, action) pairs.

    Note: these aren't on the same scale — r_hat predicts per-step reward
    while Q* is cumulative. We normalize both to zero-mean unit-variance
    before comparing, so this measures rank agreement more than absolute match.
    """
    learned = []
    optimal = []
    terminals = env.terminal_objects_placement

    for y in range(env.height):
        for x in range(env.width):
            if (x, y) in env.wall_placement or (x, y) in terminals:
                continue
            for a in range(4):
                r_hat = reward_ensemble.predict_reward([x, y], a)
                q_star = Q_star.get((x, y, a), 0.0)
                learned.append(r_hat)
                optimal.append(q_star)

    if not learned:
        return 0.0

    # Normalize both to zero-mean unit-std for fair comparison
    def normalize(vals):
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = var ** 0.5 if var > 1e-8 else 1.0
        return [(v - mean) / std for v in vals]

    learned_norm = normalize(learned)
    optimal_norm = normalize(optimal)

    mse = sum((l - o) ** 2 for l, o in zip(learned_norm, optimal_norm)) / len(learned)
    return mse


def evaluate_true_return(env, policy, terminals: dict,
                         step_penalty: float = -1.0,
                         gamma: float = 0.99,
                         n_episodes: int = 20,
                         use_distance_shaping: bool = True) -> float:
    """
    Average discounted return of the current policy under the TRUE reward
    function (not the learned one). This is the ground truth performance metric.
    """
    goal_pos = None
    max_dist = 1
    if terminals and use_distance_shaping:
        goal_pos = max(terminals, key=terminals.get)
        max_dist = abs(goal_pos[0]) + abs(goal_pos[1]) + 2

    total_return = 0.0

    for _ in range(n_episodes):
        obs = env.reset()
        episode_return = 0.0
        discount = 1.0

        while not env.done:
            action, _ = policy.get_action(obs)
            obs, done, info = env.step(action)

            pos = tuple(obs)
            if info["event"] == "terminal":
                r = float(info.get("terminal_value", 0))
            elif use_distance_shaping and goal_pos is not None:
                dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                r = step_penalty + (max_dist - dist) / max_dist
            else:
                r = step_penalty

            episode_return += discount * r
            discount *= gamma

        total_return += episode_return

    return total_return / n_episodes


def optimal_return_from_start(V_star: dict, start: tuple = (0, 0)) -> float:
    """V*(start) — the best achievable discounted return from the start state."""
    return V_star.get(start, 0.0)


def render_policy_grid(env, policy_dict: dict) -> str:
    """
    Render a policy (dict: (x,y)->action_int) as an arrow grid string.
    Works for both optimal policy dicts and learned policy dicts.
    """
    arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    lines = []
    for y in range(env.height):
        row = []
        for x in range(env.width):
            if (x, y) in env.wall_placement:
                row.append("#")
            elif (x, y) in env.terminal_objects_placement:
                val = env.terminal_objects_placement[(x, y)]
                row.append("G" if val > 0 else "T")
            elif (x, y) in policy_dict:
                row.append(arrows.get(policy_dict[(x, y)], "?"))
            else:
                row.append(".")
        lines.append(" ".join(row))
    return "\n".join(lines)


def get_learned_policy_dict(policy, env) -> dict:
    """Convert the neural network policy to a dict: (x,y) -> best action int."""
    result = {}
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) in env.wall_placement:
                continue
            probs = policy.get_action_probabilities([x, y])
            result[(x, y)] = probs.index(max(probs))
    return result


def compute_baseline(env, step_penalty: float = -1.0, gamma: float = 0.99,
                     use_distance_shaping: bool = True):
    """
    Convenience function: compute all DP baselines at once.
    Returns (V_star, Q_star, pi_star, optimal_return).
    """
    terminals = env.terminal_objects_placement
    V_star = value_iteration(env, terminals, step_penalty, gamma,
                             use_distance_shaping)
    Q_star = compute_q_values(env, V_star, terminals, step_penalty, gamma,
                              use_distance_shaping)
    pi_star = get_optimal_policy(Q_star, env)
    opt_return = optimal_return_from_start(V_star, (0, 0))
    return V_star, Q_star, pi_star, opt_return
