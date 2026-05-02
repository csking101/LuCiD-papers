"""
Tests for the GridWorld environment.
=====================================

Covers:
- Grid construction and wall placement
- Observation normalisation
- Movement mechanics (valid moves, wall collision, boundary)
- Reward shaping
- Episode termination (goal and timeout)
- BFS shortest path
- Trajectory rollouts (random and policy-driven)
- Trajectory dataclass properties
- Environment copy isolation
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow imports from the adventure directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import (
    ACTION_DELTAS,
    COIN,
    DOWN,
    EMPTY,
    GEM,
    LEFT,
    NUM_ACTIONS,
    PICKUP_VALUES,
    RIGHT,
    SIZE,
    UP,
    WALL,
    GridWorld,
    Trajectory,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
@pytest.fixture
def env():
    """Default 8x8 environment with standard walls."""
    return GridWorld()


@pytest.fixture
def tiny_env():
    """Minimal 4x4 environment with no walls or pickups for deterministic testing."""
    return GridWorld(size=4, walls=[], pickups=[], max_steps=50)


@pytest.fixture
def corridor_env():
    """5x5 grid with a single corridor forcing one path."""
    #  S . . . .
    #  # # # # .
    #  . . . . .
    #  . # # # #
    #  . . . . G
    walls = [
        (1, 0), (1, 1), (1, 2), (1, 3),
        (3, 1), (3, 2), (3, 3), (3, 4),
    ]
    return GridWorld(size=5, walls=walls, pickups=[], max_steps=50)


# -----------------------------------------------------------------------
# Grid construction
# -----------------------------------------------------------------------
class TestGridConstruction:
    def test_default_size(self, env):
        assert env.size == SIZE == 8
        assert env.grid.shape == (8, 8)

    def test_start_and_goal_are_empty(self, env):
        assert env.grid[env.start] == EMPTY
        assert env.grid[env.goal] == EMPTY

    def test_walls_placed_correctly(self, env):
        """Row 2 cols 2-5 should be walls in the 8x8 layout."""
        for c in range(2, 6):
            assert env.grid[2, c] == WALL, f"Expected wall at (2, {c})"
        # Gaps at col 0-1 and 6-7
        assert env.grid[2, 0] == EMPTY
        assert env.grid[2, 1] == EMPTY
        assert env.grid[2, 6] == EMPTY
        assert env.grid[2, 7] == EMPTY

    def test_custom_walls(self):
        walls = [(0, 1), (1, 0)]
        e = GridWorld(size=3, walls=walls)
        assert e.grid[0, 1] == WALL
        assert e.grid[1, 0] == WALL
        assert e.grid[0, 0] == EMPTY  # start always clear

    def test_no_walls(self):
        e = GridWorld(size=5, walls=[])
        assert np.all(e.grid == EMPTY)

    def test_out_of_bounds_walls_ignored(self):
        """Walls outside the grid should not crash."""
        e = GridWorld(size=3, walls=[(99, 99), (-1, -1)])
        assert e.grid.shape == (3, 3)


# -----------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------
class TestObservation:
    def test_initial_obs(self, env):
        obs = env.reset()
        np.testing.assert_array_almost_equal(obs, [0.0, 0.0])

    def test_obs_at_goal(self, tiny_env):
        tiny_env.pos = tiny_env.goal
        obs = tiny_env._obs()
        np.testing.assert_array_almost_equal(obs, [1.0, 1.0])

    def test_obs_dtype(self, env):
        obs = env.reset()
        assert obs.dtype == np.float32

    def test_obs_shape(self, env):
        obs = env.reset()
        assert obs.shape == (2,)

    def test_obs_mid_grid(self):
        e = GridWorld(size=5, walls=[])
        e.reset()
        e.pos = (2, 3)
        obs = e._obs()
        np.testing.assert_array_almost_equal(obs, [2 / 4, 3 / 4])


# -----------------------------------------------------------------------
# Movement
# -----------------------------------------------------------------------
class TestMovement:
    def test_move_right(self, tiny_env):
        tiny_env.reset()
        obs, reward, done = tiny_env.step(RIGHT)
        assert tiny_env.pos == (0, 1)

    def test_move_down(self, tiny_env):
        tiny_env.reset()
        obs, reward, done = tiny_env.step(DOWN)
        assert tiny_env.pos == (1, 0)

    def test_wall_collision_stays_in_place(self, env):
        """Walking into a wall should not change position."""
        env.reset()
        # Place agent next to a wall.  Row 2, col 2 is a wall in 8x8 layout.
        # Position (1, 2), moving DOWN should hit wall at (2, 2).
        env.pos = (1, 2)
        old_pos = env.pos
        env.step(DOWN)
        assert env.pos == old_pos

    def test_boundary_collision_north(self, tiny_env):
        tiny_env.reset()
        tiny_env.pos = (0, 0)
        tiny_env.step(UP)
        assert tiny_env.pos == (0, 0)

    def test_boundary_collision_west(self, tiny_env):
        tiny_env.reset()
        tiny_env.pos = (0, 0)
        tiny_env.step(LEFT)
        assert tiny_env.pos == (0, 0)

    def test_boundary_collision_south(self, tiny_env):
        tiny_env.reset()
        tiny_env.pos = (3, 0)
        tiny_env.step(DOWN)
        assert tiny_env.pos == (3, 0)

    def test_boundary_collision_east(self, tiny_env):
        tiny_env.reset()
        tiny_env.pos = (0, 3)
        tiny_env.step(RIGHT)
        assert tiny_env.pos == (0, 3)

    def test_path_tracking(self, tiny_env):
        tiny_env.reset()
        tiny_env.step(RIGHT)
        tiny_env.step(DOWN)
        assert tiny_env.path == [(0, 0), (0, 1), (1, 1)]

    def test_invalid_action_raises(self, env):
        env.reset()
        with pytest.raises(AssertionError):
            env.step(-1)
        with pytest.raises(AssertionError):
            env.step(4)


# -----------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------
class TestReward:
    def test_goal_reward(self, tiny_env):
        tiny_env.reset()
        tiny_env.pos = (3, 2)  # one step left of goal (3,3)
        _, reward, done = tiny_env.step(RIGHT)
        assert reward == 10.0
        assert done is True

    def test_step_penalty(self, tiny_env):
        tiny_env.reset()
        _, reward, done = tiny_env.step(RIGHT)  # just a normal step
        assert reward == pytest.approx(-0.01)

    def test_done_step_returns_zero_reward(self, tiny_env):
        tiny_env.reset()
        tiny_env.done = True
        _, reward, done = tiny_env.step(RIGHT)
        assert reward == 0.0
        assert done is True


# -----------------------------------------------------------------------
# Episode termination
# -----------------------------------------------------------------------
class TestTermination:
    def test_goal_terminates(self, tiny_env):
        tiny_env.reset()
        # Navigate to goal: right x3, down x3
        for _ in range(3):
            tiny_env.step(RIGHT)
        for _ in range(2):
            _, _, done = tiny_env.step(DOWN)
            assert done is False
        _, _, done = tiny_env.step(DOWN)
        assert done is True
        assert tiny_env.pos == tiny_env.goal

    def test_timeout_terminates(self):
        e = GridWorld(size=4, walls=[], pickups=[], max_steps=3)
        e.reset()
        e.step(RIGHT)
        e.step(RIGHT)
        _, _, done = e.step(RIGHT)
        assert done is True
        # Agent didn't reach goal
        assert e.pos != e.goal

    def test_steps_counted(self, tiny_env):
        tiny_env.reset()
        for _ in range(5):
            tiny_env.step(RIGHT)
        assert tiny_env.steps == 5


# -----------------------------------------------------------------------
# Valid actions
# -----------------------------------------------------------------------
class TestValidActions:
    def test_corner_start(self, env):
        env.reset()
        valid = env.valid_actions()
        # At (0,0) can only go RIGHT and DOWN
        assert set(valid) == {DOWN, RIGHT}

    def test_corner_goal(self, env):
        env.reset()
        env.pos = env.goal  # (7,7)
        valid = env.valid_actions()
        assert set(valid) == {UP, LEFT}

    def test_open_center(self):
        e = GridWorld(size=5, walls=[])
        e.reset()
        e.pos = (2, 2)
        valid = e.valid_actions()
        assert set(valid) == {UP, DOWN, LEFT, RIGHT}

    def test_adjacent_to_wall(self, env):
        """At (1, 2), DOWN goes to wall (2, 2)."""
        env.reset()
        valid = env.valid_actions(pos=(1, 2))
        assert DOWN not in valid

    def test_custom_position(self, env):
        env.reset()
        valid = env.valid_actions(pos=(0, 5))
        assert UP not in valid  # top edge


# -----------------------------------------------------------------------
# BFS shortest path
# -----------------------------------------------------------------------
class TestShortestPath:
    def test_exists_default_grid(self, env):
        path = env.shortest_path()
        assert path is not None
        assert path[0] == env.start
        assert path[-1] == env.goal

    def test_trivial_path(self):
        e = GridWorld(size=2, walls=[])
        path = e.shortest_path()
        assert path is not None
        assert len(path) == 3  # (0,0) → (0,1)/(1,0) → (1,1)

    def test_no_path(self):
        """Completely walled-off goal."""
        walls = [(0, 1), (1, 0)]  # blocks (0,0) in a 2x2
        e = GridWorld(size=2, walls=walls)
        path = e.shortest_path()
        assert path is None

    def test_start_equals_goal(self):
        e = GridWorld(size=3, walls=[])
        path = e.shortest_path(start=(1, 1), goal=(1, 1))
        assert path == [(1, 1)]

    def test_corridor_single_path(self, corridor_env):
        path = corridor_env.shortest_path()
        assert path is not None
        # In the corridor env there's only one path
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)

    def test_all_positions_reachable_default(self, env):
        """Every non-wall cell should be reachable from start."""
        for r in range(env.size):
            for c in range(env.size):
                if env.grid[r, c] != WALL:
                    path = env.shortest_path(goal=(r, c))
                    assert path is not None, f"Cell ({r},{c}) unreachable"


# -----------------------------------------------------------------------
# Trajectory dataclass
# -----------------------------------------------------------------------
class TestTrajectory:
    def test_total_reward(self):
        t = Trajectory(rewards=[1.0, 2.0, -0.5])
        assert t.total_reward == pytest.approx(2.5)

    def test_length(self):
        t = Trajectory(actions=[0, 1, 2, 3])
        assert t.length == 4

    def test_num_turns(self):
        t = Trajectory(actions=[RIGHT, RIGHT, DOWN, DOWN, RIGHT])
        assert t.num_turns == 2  # RIGHT→DOWN, DOWN→RIGHT

    def test_num_turns_single_action(self):
        t = Trajectory(actions=[UP])
        assert t.num_turns == 0

    def test_num_turns_empty(self):
        t = Trajectory()
        assert t.num_turns == 0

    def test_unique_cells(self):
        t = Trajectory(positions=[(0, 0), (0, 1), (0, 0), (0, 1), (0, 2)])
        assert t.unique_cells == 3

    def test_state_tensor(self):
        t = Trajectory(
            states=[np.array([0.0, 0.0]), np.array([0.1, 0.2])]
        )
        arr = t.state_tensor()
        assert arr.dtype == np.float32
        assert arr.shape == (2, 2)


# -----------------------------------------------------------------------
# Rollouts
# -----------------------------------------------------------------------
class TestRollout:
    def test_random_rollout_terminates(self, env):
        np.random.seed(42)
        traj = env.rollout_random()
        assert traj.done is True
        assert traj.length > 0

    def test_random_rollout_positions(self, env):
        np.random.seed(42)
        traj = env.rollout_random()
        # positions has one more entry than actions (includes final pos)
        assert len(traj.positions) == len(traj.actions) + 1
        assert traj.positions[0] == env.start

    def test_policy_rollout_reaches_goal(self, tiny_env):
        """A deterministic policy that goes right then down should reach goal."""
        call_count = [0]

        def smart_policy(obs):
            call_count[0] += 1
            r, c = obs[0] * 3, obs[1] * 3  # denormalise for 4x4
            if c < 3:
                return RIGHT, 0.0, 0.0
            else:
                return DOWN, 0.0, 0.0

        traj = tiny_env.rollout(smart_policy)
        assert traj.reached_goal is True
        assert traj.positions[-1] == tiny_env.goal

    def test_rollout_max_steps_override(self, tiny_env):
        traj = tiny_env.rollout_random(max_steps=5)
        assert traj.length <= 5
        # max_steps should be restored after rollout
        assert tiny_env.max_steps == 50

    def test_rollout_log_probs_and_values(self, tiny_env):
        def policy_with_extras(obs):
            return RIGHT, -0.5, 3.14

        traj = tiny_env.rollout(policy_with_extras, max_steps=3)
        assert all(lp == pytest.approx(-0.5) for lp in traj.log_probs)
        assert all(v == pytest.approx(3.14) for v in traj.values)


# -----------------------------------------------------------------------
# Environment copy
# -----------------------------------------------------------------------
class TestCopy:
    def test_copy_is_independent(self, env):
        env.reset()
        env.step(RIGHT)
        env_copy = env.copy()

        # Mutate original
        env.step(DOWN)

        # Copy should be unchanged
        assert env_copy.pos == (0, 1)
        assert env.pos != env_copy.pos

    def test_copy_grid_independent(self, env):
        env_copy = env.copy()
        env_copy.grid[0, 1] = WALL
        assert env.grid[0, 1] != WALL


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------
class TestEdgeCases:
    def test_size_one_grid(self):
        """1x1 grid: start == goal, immediate done."""
        e = GridWorld(size=1, walls=[])
        obs = e.reset()
        # Already at goal, but done isn't set until a step
        np.testing.assert_array_almost_equal(obs, [0.0, 0.0])

    def test_size_two_grid(self):
        e = GridWorld(size=2, walls=[])
        path = e.shortest_path()
        assert path is not None
        assert len(path) == 3  # start + mid + goal

    def test_many_rollouts_dont_leak_state(self, env):
        """Running many rollouts shouldn't accumulate state."""
        for _ in range(20):
            traj = env.rollout_random(max_steps=10)
        assert env.steps == 0 or env.done  # rollout resets internally


# -----------------------------------------------------------------------
# Pickups (coins & gems)
# -----------------------------------------------------------------------
class TestPickups:
    def test_default_pickups_placed(self, env):
        """Default 8x8 env should have pickups from _DEFAULT_PICKUPS."""
        assert len(env.initial_pickups) > 0
        # Should have both coins and gems
        types = set(env.initial_pickups.values())
        assert COIN in types
        assert GEM in types

    def test_no_pickup_on_wall_or_start_or_goal(self, env):
        """Pickups should never overlap with walls, start, or goal."""
        for pos, ptype in env.initial_pickups.items():
            assert env.grid[pos] == EMPTY, f"Pickup at {pos} overlaps wall"
            assert pos != env.start, "Pickup at start"
            assert pos != env.goal, "Pickup at goal"

    def test_custom_no_pickups(self):
        """pickups=[] should produce an env with no pickups."""
        e = GridWorld(size=4, walls=[], pickups=[])
        assert len(e.initial_pickups) == 0
        assert len(e.active_pickups) == 0

    def test_custom_pickups(self):
        pickups = [(1, 1, COIN), (2, 2, GEM)]
        e = GridWorld(size=4, walls=[], pickups=pickups)
        assert len(e.initial_pickups) == 2
        assert e.initial_pickups[(1, 1)] == COIN
        assert e.initial_pickups[(2, 2)] == GEM

    def test_pickup_on_wall_ignored(self):
        """A pickup placed on a wall cell should be silently ignored."""
        walls = [(1, 1)]
        pickups = [(1, 1, COIN)]  # overlaps wall
        e = GridWorld(size=4, walls=walls, pickups=pickups)
        assert (1, 1) not in e.initial_pickups

    def test_coin_gives_bonus(self):
        """Stepping onto a coin should add PICKUP_VALUES[COIN] to reward."""
        pickups = [(0, 1, COIN)]
        e = GridWorld(size=4, walls=[], pickups=pickups)
        e.reset()
        _, reward, _ = e.step(RIGHT)  # move to (0,1) — the coin
        # reward = step_penalty (-0.01) + coin bonus (0.5)
        assert reward == pytest.approx(-0.01 + PICKUP_VALUES[COIN])

    def test_gem_gives_bonus(self):
        """Stepping onto a gem should add PICKUP_VALUES[GEM] to reward."""
        pickups = [(0, 1, GEM)]
        e = GridWorld(size=4, walls=[], pickups=pickups)
        e.reset()
        _, reward, _ = e.step(RIGHT)
        assert reward == pytest.approx(-0.01 + PICKUP_VALUES[GEM])

    def test_pickup_disappears_after_collection(self):
        """After collecting a pickup, revisiting the cell gives no bonus."""
        pickups = [(0, 1, COIN)]
        e = GridWorld(size=4, walls=[], pickups=pickups)
        e.reset()
        _, r1, _ = e.step(RIGHT)  # collect coin
        assert r1 == pytest.approx(-0.01 + PICKUP_VALUES[COIN])
        e.step(LEFT)   # back to (0,0)
        _, r2, _ = e.step(RIGHT)  # revisit (0,1) — no bonus
        assert r2 == pytest.approx(-0.01)

    def test_reset_restores_pickups(self):
        """After reset, all pickups should be active again."""
        pickups = [(0, 1, COIN), (1, 0, GEM)]
        e = GridWorld(size=4, walls=[], pickups=pickups)
        e.reset()
        e.step(RIGHT)  # collect coin at (0,1)
        assert len(e.active_pickups) == 1
        e.reset()
        assert len(e.active_pickups) == 2

    def test_collected_pickups_tracked(self):
        """env.collected_pickups should record each collection."""
        pickups = [(0, 1, COIN), (0, 2, GEM)]
        e = GridWorld(size=4, walls=[], pickups=pickups)
        e.reset()
        e.step(RIGHT)  # collect coin at (0,1)
        e.step(RIGHT)  # collect gem at (0,2)
        assert len(e.collected_pickups) == 2
        assert e.collected_pickups[0] == (0, 1, COIN)
        assert e.collected_pickups[1] == (0, 2, GEM)

    def test_trajectory_records_pickups(self):
        """Rollout trajectory should capture collected pickups."""
        pickups = [(0, 1, COIN), (0, 2, GEM)]
        e = GridWorld(size=4, walls=[], pickups=pickups, max_steps=10)

        def go_right(obs):
            return RIGHT, 0.0, 0.0

        traj = e.rollout(go_right, max_steps=6)
        # Agent goes right each step: (0,0)→(0,1)→(0,2)→(0,3)→stays at (0,3)
        assert len(traj.pickups_collected) == 2
        assert traj.pickup_reward == pytest.approx(
            PICKUP_VALUES[COIN] + PICKUP_VALUES[GEM]
        )

    def test_pickup_reward_property(self):
        t = Trajectory(pickups_collected=[(0, 1, COIN), (2, 2, GEM)])
        assert t.pickup_reward == pytest.approx(
            PICKUP_VALUES[COIN] + PICKUP_VALUES[GEM]
        )

    def test_pickup_reward_empty(self):
        t = Trajectory()
        assert t.pickup_reward == pytest.approx(0.0)

    def test_bfs_ignores_pickups(self, env):
        """Pickups should not block BFS shortest path."""
        path = env.shortest_path()
        assert path is not None
        assert path[0] == env.start
        assert path[-1] == env.goal

    def test_all_pickup_positions_reachable(self, env):
        """Every pickup should be reachable from start."""
        for pos in env.initial_pickups:
            path = env.shortest_path(goal=pos)
            assert path is not None, f"Pickup at {pos} unreachable"
