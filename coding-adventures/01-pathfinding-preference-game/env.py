"""
Grid World Environment for RLHF Path-Finding
=============================================

An 8x8 grid with obstacles that create **three distinct corridors** from
start (top-left) to goal (bottom-right), plus **collectible pickups** (coins
and gems) that reward exploration.  Multiple viable routes and pickup
placement tradeoffs ensure that human preferences actually matter — do you
prefer the shortest path, or a detour that collects a rare gem?

Observation : np.array([row/(size-1), col/(size-1)])  — normalised position
Actions     : 0=UP  1=DOWN  2=LEFT  3=RIGHT
Pickups     : COIN (+0.5), GEM (+2.0) — disappear on collection

LLM parallel
------------
The grid world is to the path-finding agent what the vocabulary and context
window are to a language model: the space of possible outputs (paths / token
sequences) that the policy must learn to navigate.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMPTY = 0
WALL = 1
COIN = 2
GEM = 3
SIZE = 8

# Pickup rewards — coins are common, gems are rare and valuable
PICKUP_VALUES: Dict[int, float] = {COIN: 0.5, GEM: 2.0}

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_ARROWS = ["\u2191", "\u2193", "\u2190", "\u2192"]  # ↑ ↓ ← →
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NUM_ACTIONS = 4

# Default obstacle layout — three corridors (8×8)
# Top corridor:    rows 0-1   (gap at cols 0-1 and 6-7 in wall at row 2)
# Middle corridor: rows 3-4   (vertical barrier at col 3 forces detour)
# Bottom corridor: rows 5-7   (gap at cols 0-1 and 6-7 in wall at row 5)
_DEFAULT_WALLS: List[Tuple[int, int]] = [
    # Horizontal wall across row 2, gaps at col 0-1 and col 6-7
    *[(2, c) for c in range(2, 6)],
    # Horizontal wall across row 5, gaps at col 0-1 and col 6-7
    *[(5, c) for c in range(2, 6)],
    # Vertical barrier in middle corridor at col 3, rows 3-4
    (3, 3), (4, 3),
    # Texture obstacles (make paths interesting, not blocking)
    (1, 4),
    (4, 1), (4, 5),
    (6, 3),
]

# Default pickup layout — coins along corridors, gems at detour spots
# Format: (row, col, type)
_DEFAULT_PICKUPS: List[Tuple[int, int, int]] = [
    # Coins (+0.5) — on natural paths
    (0, 4, COIN),   # top corridor middle
    (1, 1, COIN),   # near start
    (3, 1, COIN),   # middle corridor left
    (3, 5, COIN),   # middle corridor right (past barrier)
    (4, 6, COIN),   # middle right near gap
    (6, 1, COIN),   # bottom corridor left
    (7, 3, COIN),   # bottom near goal
    # Gems (+2.0) — require detours
    (0, 7, GEM),    # far top-right corner
    (3, 6, GEM),    # middle-right corridor
    (7, 0, GEM),    # bottom-left corner (big detour)
]


# ---------------------------------------------------------------------------
# Trajectory data
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    """Stores a complete trajectory through the grid."""

    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    positions: List[Tuple[int, int]] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    done: bool = False
    reached_goal: bool = False
    pickups_collected: List[Tuple[int, int, int]] = field(default_factory=list)
    """List of (row, col, type) for each pickup collected during this episode."""

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def pickup_reward(self) -> float:
        """Total bonus reward from collected pickups."""
        return sum(PICKUP_VALUES.get(t, 0.0) for _, _, t in self.pickups_collected)

    @property
    def num_turns(self) -> int:
        """Count direction changes in the trajectory."""
        if len(self.actions) < 2:
            return 0
        return sum(
            1 for a, b in zip(self.actions[:-1], self.actions[1:]) if a != b
        )

    @property
    def unique_cells(self) -> int:
        return len(set(self.positions))

    def state_tensor(self) -> np.ndarray:
        """Return states as a float32 numpy array (N, 2)."""
        if len(self.states) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(self.states, dtype=np.float32)


# ---------------------------------------------------------------------------
# Grid World
# ---------------------------------------------------------------------------
class GridWorld:
    """
    8x8 grid world with configurable obstacles and collectible pickups.

    Parameters
    ----------
    size : int
        Side length of the square grid.
    walls : list of (row, col) or None
        Obstacle positions.  ``None`` uses the default three-corridor layout.
    pickups : list of (row, col, type) or None
        Pickup positions and types (COIN or GEM).  ``None`` uses the default
        layout.  Pass ``[]`` for no pickups.
    max_steps : int
        Maximum steps per episode before forced termination.
    """

    def __init__(
        self,
        size: int = SIZE,
        walls: Optional[List[Tuple[int, int]]] = None,
        pickups: Optional[List[Tuple[int, int, int]]] = None,
        max_steps: int = 200,
    ):
        self.size = size
        self.max_steps = max_steps
        self.start: Tuple[int, int] = (0, 0)
        self.goal: Tuple[int, int] = (size - 1, size - 1)

        # Build grid
        self.grid = np.zeros((size, size), dtype=np.int8)
        wall_list = walls if walls is not None else _DEFAULT_WALLS
        for r, c in wall_list:
            if 0 <= r < size and 0 <= c < size:
                self.grid[r, c] = WALL

        # Ensure start and goal are clear
        self.grid[self.start] = EMPTY
        self.grid[self.goal] = EMPTY

        # Pickups — stored separately so the base grid stays walls/empty only
        pickup_list = pickups if pickups is not None else _DEFAULT_PICKUPS
        self.initial_pickups: Dict[Tuple[int, int], int] = {}
        for r, c, ptype in pickup_list:
            if (
                0 <= r < size
                and 0 <= c < size
                and self.grid[r, c] == EMPTY
                and (r, c) != self.start
                and (r, c) != self.goal
            ):
                self.initial_pickups[(r, c)] = ptype
        self.active_pickups: Dict[Tuple[int, int], int] = dict(self.initial_pickups)

        # Episode state
        self.pos: Tuple[int, int] = self.start
        self.steps: int = 0
        self.done: bool = False
        self.path: List[Tuple[int, int]] = [self.start]
        self.collected_pickups: List[Tuple[int, int, int]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation."""
        self.pos = self.start
        self.steps = 0
        self.done = False
        self.path = [self.start]
        self.active_pickups = dict(self.initial_pickups)
        self.collected_pickups = []
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take one step in the environment.

        Returns
        -------
        obs : np.ndarray   — normalised (row, col)
        reward : float      — shaped reward + any pickup bonus
        done : bool         — True if goal reached or time-out
        """
        assert 0 <= action < NUM_ACTIONS, f"Invalid action {action}"
        if self.done:
            return self._obs(), 0.0, True

        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc

        # Only move if in-bounds and not a wall
        if self._in_bounds(nr, nc) and self.grid[nr, nc] != WALL:
            self.pos = (nr, nc)

        self.steps += 1
        self.path.append(self.pos)

        reached_goal = self.pos == self.goal
        out_of_time = self.steps >= self.max_steps
        self.done = reached_goal or out_of_time

        reward = self._shaped_reward(reached_goal)

        # Pickup collection — consumed on first visit
        if self.pos in self.active_pickups:
            ptype = self.active_pickups.pop(self.pos)
            bonus = PICKUP_VALUES.get(ptype, 0.0)
            reward += bonus
            self.collected_pickups.append((self.pos[0], self.pos[1], ptype))

        return self._obs(), reward, self.done

    # ------------------------------------------------------------------
    # Reward (used only during pre-training / Phase 1)
    # ------------------------------------------------------------------
    def _shaped_reward(self, reached_goal: bool) -> float:
        """Simple hand-coded reward for pre-training."""
        if reached_goal:
            return 10.0
        return -0.01  # small step penalty encourages efficiency

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        return np.array(
            [self.pos[0] / max(self.size - 1, 1),
             self.pos[1] / max(self.size - 1, 1)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def is_wall(self, r: int, c: int) -> bool:
        return bool(self.grid[r, c] == WALL)

    def valid_actions(self, pos: Optional[Tuple[int, int]] = None) -> List[int]:
        """Return actions that do not walk into a wall or out of bounds."""
        p = pos or self.pos
        valid = []
        for a, (dr, dc) in enumerate(ACTION_DELTAS):
            nr, nc = p[0] + dr, p[1] + dc
            if self._in_bounds(nr, nc) and self.grid[nr, nc] != WALL:
                valid.append(a)
        return valid

    def copy(self) -> "GridWorld":
        """Deep copy the environment (useful for parallel rollouts)."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # BFS shortest path (for reference / evaluation)
    # ------------------------------------------------------------------
    def shortest_path(
        self,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        BFS from *start* to *goal*.  Returns the path as a list of
        positions, or ``None`` if no path exists.
        """
        s = start or self.start
        g = goal or self.goal

        if s == g:
            return [s]

        visited = {s}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        queue: deque[Tuple[int, int]] = deque([s])

        while queue:
            r, c = queue.popleft()
            for dr, dc in ACTION_DELTAS:
                nr, nc = r + dr, c + dc
                if (
                    self._in_bounds(nr, nc)
                    and self.grid[nr, nc] != WALL
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    if (nr, nc) == g:
                        # Reconstruct
                        path = [(nr, nc)]
                        while path[-1] != s:
                            path.append(parent[path[-1]])
                        return list(reversed(path))
                    queue.append((nr, nc))
        return None  # no path

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------
    PolicyFn = Callable[[np.ndarray], Tuple[int, float, float]]

    def rollout(
        self,
        policy_fn: "GridWorld.PolicyFn",
        max_steps: Optional[int] = None,
    ) -> Trajectory:
        """
        Run one full episode using ``policy_fn(obs) -> (action, log_prob, value)``.

        Returns a :class:`Trajectory` dataclass.
        """
        ms = max_steps or self.max_steps
        old_max = self.max_steps
        self.max_steps = ms

        traj = Trajectory()
        obs = self.reset()

        while not self.done:
            action, log_prob, value = policy_fn(obs)
            traj.states.append(obs.copy())
            traj.positions.append(self.pos)
            traj.actions.append(int(action))
            traj.log_probs.append(float(log_prob))
            traj.values.append(float(value))

            obs, reward, done = self.step(action)
            traj.rewards.append(reward)

        traj.done = True
        traj.reached_goal = self.pos == self.goal
        traj.pickups_collected = list(self.collected_pickups)
        # Record final position (one more position than actions)
        traj.positions.append(self.pos)

        self.max_steps = old_max
        return traj

    def rollout_random(self, max_steps: Optional[int] = None) -> Trajectory:
        """Generate a trajectory with a uniform random policy."""

        def _random_policy(_obs: np.ndarray) -> Tuple[int, float, float]:
            action = np.random.randint(NUM_ACTIONS)
            return action, 0.0, 0.0

        return self.rollout(_random_policy, max_steps=max_steps)
