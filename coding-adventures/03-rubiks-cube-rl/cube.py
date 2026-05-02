"""2x2 Pocket Cube environment for RL training.

State representation:
    24 integers (0-5), one per sticker.
    Indices 0-3: U face (color 0 = White when solved)
    Indices 4-7: D face (color 1 = Yellow)
    Indices 8-11: F face (color 2 = Green)
    Indices 12-15: B face (color 3 = Blue)
    Indices 16-19: L face (color 4 = Orange)
    Indices 20-23: R face (color 5 = Red)

    Within each face (viewed from outside the cube):
        0 1
        2 3

    Unfolded layout with absolute indices:
                 0   1
                 2   3
        16  17 | 8   9 | 20  21 | 12  13
        18  19 |10  11 | 22  23 | 14  15
                 4   5
                 6   7

Actions:
    0 = U  (Up clockwise, viewed from above)
    1 = U' (Up counter-clockwise)
    2 = R  (Right clockwise, viewed from the right)
    3 = R' (Right counter-clockwise)
    4 = F  (Front clockwise, viewed from front)
    5 = F' (Front counter-clockwise)

    The DLB (Down-Left-Back) corner is fixed. Its stickers at
    indices 6 (D), 18 (L), 15 (B) never move. This removes
    rotational equivalence, which is standard for the 2x2 cube.

    These 3 generators (U, R, F) and their inverses are sufficient
    to reach all 3,674,160 reachable states.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch

# ─── Constants ───────────────────────────────────────────────────────────────

NUM_FACES = 6
STICKERS_PER_FACE = 4
NUM_STICKERS = NUM_FACES * STICKERS_PER_FACE  # 24
NUM_COLORS = 6
STATE_DIM = NUM_STICKERS * NUM_COLORS  # 144 (one-hot)
NUM_ACTIONS = 6

# Face indices
U, D, F, B, L, R = 0, 1, 2, 3, 4, 5
FACE_NAMES = ["U", "D", "F", "B", "L", "R"]
FACE_COLORS = ["white", "yellow", "green", "blue", "orange", "red"]

# Action names
ACTION_NAMES = ["U", "U'", "R", "R'", "F", "F'"]

# Inverse action mapping: action_index -> inverse_action_index
INVERSE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

# God's number for 2x2 in quarter-turn metric
GODS_NUMBER = 14

# DLB corner sticker indices (these never move)
DLB_INDICES = (6, 18, 15)

# ─── Move definitions ────────────────────────────────────────────────────────
#
# Each move is a list of 4-cycles.  A cycle (a, b, c, d) means:
#     a → b,  b → c,  c → d,  d → a
#
# Derived by tracing cubie movements through the 8-corner model.
# Verified: each move applied 4 times = identity; move + inverse = identity.

MOVE_CYCLES: dict[int, list[tuple[int, int, int, int]]] = {
    # U  (Up face clockwise, viewed from above)
    0: [(0, 1, 3, 2), (16, 12, 20, 8), (13, 21, 9, 17)],
    # U' (inverse)
    1: [(0, 2, 3, 1), (16, 8, 20, 12), (13, 17, 9, 21)],
    # R  (Right face clockwise, viewed from the right)
    2: [(20, 21, 23, 22), (3, 12, 7, 11), (9, 1, 14, 5)],
    # R' (inverse)
    3: [(20, 22, 23, 21), (3, 11, 7, 12), (9, 5, 14, 1)],
    # F  (Front face clockwise, viewed from the front)
    4: [(8, 9, 11, 10), (2, 20, 5, 19), (17, 3, 22, 4)],
    # F' (inverse)
    5: [(8, 10, 11, 9), (2, 19, 5, 20), (17, 4, 22, 3)],
}

# Solved state: face i has color i for all stickers
SOLVED_STATE: list[int] = [i // STICKERS_PER_FACE for i in range(NUM_STICKERS)]


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of a single environment step."""

    state: list[int]
    reward: float
    done: bool
    info: dict


# ─── Pocket Cube ──────────────────────────────────────────────────────────────

class PocketCube:
    """2x2 Pocket Cube RL environment.

    Observation: 144-dim one-hot tensor (24 stickers × 6 colors).
    Action space: 6 discrete actions (U, U', R, R', F, F').
    Reward: +1.0 on solve, -0.02 per step otherwise.
    Termination: solved or max_steps reached.
    """

    def __init__(self) -> None:
        self.state: list[int] = list(SOLVED_STATE)
        self._steps: int = 0
        self._max_steps: int = 20
        self._scramble_depth: int = 0
        self._scramble_moves: list[int] = []

    # ── Core interface ────────────────────────────────────────────────────

    def reset(self) -> list[int]:
        """Reset to solved state. Returns the state."""
        self.state = list(SOLVED_STATE)
        self._steps = 0
        self._scramble_depth = 0
        self._scramble_moves = []
        return list(self.state)

    def is_solved(self) -> bool:
        """True if every face has all 4 stickers the same colour."""
        for i in range(0, NUM_STICKERS, STICKERS_PER_FACE):
            face = self.state[i : i + STICKERS_PER_FACE]
            if face[0] != face[1] or face[0] != face[2] or face[0] != face[3]:
                return False
        return True

    def apply_move(self, action: int) -> None:
        """Apply a move (0-5) to the current state in-place."""
        if action < 0 or action >= NUM_ACTIONS:
            raise ValueError(f"Invalid action {action}, must be 0-{NUM_ACTIONS - 1}")
        cycles = MOVE_CYCLES[action]
        new_state = list(self.state)
        for cycle in cycles:
            temp = self.state[cycle[-1]]
            for i in range(len(cycle) - 1, 0, -1):
                new_state[cycle[i]] = self.state[cycle[i - 1]]
            new_state[cycle[0]] = temp
        self.state = new_state

    def step(self, action: int) -> StepResult:
        """Apply action, return StepResult(state, reward, done, info)."""
        self.apply_move(action)
        self._steps += 1

        solved = self.is_solved()
        done = solved or self._steps >= self._max_steps

        reward = 1.0 if solved else -0.02

        return StepResult(
            state=list(self.state),
            reward=reward,
            done=done,
            info={"solved": solved, "steps": self._steps},
        )

    # ── Scramble ──────────────────────────────────────────────────────────

    def scramble(
        self, n_moves: int, rng: Optional[random.Random] = None
    ) -> list[int]:
        """Scramble the cube with *n_moves* random moves.

        Avoids applying the same face twice in a row (which would be
        equivalent to a half-turn or identity).

        Returns the list of action indices applied.
        """
        if rng is None:
            rng = random.Random()

        moves: list[int] = []
        prev_face = -1
        for _ in range(n_moves):
            while True:
                action = rng.randint(0, NUM_ACTIONS - 1)
                this_face = action // 2
                if this_face != prev_face:
                    break
            self.apply_move(action)
            moves.append(action)
            prev_face = this_face

        self._scramble_depth = n_moves
        self._scramble_moves = list(moves)
        self._max_steps = n_moves * 3 + 2
        return moves

    # ── State representations ─────────────────────────────────────────────

    def to_tensor(self) -> torch.Tensor:
        """One-hot encode the state as a 144-dim float tensor."""
        t = torch.zeros(STATE_DIM)
        for i in range(NUM_STICKERS):
            t[i * NUM_COLORS + self.state[i]] = 1.0
        return t

    def render_faces(self) -> dict[str, list[list[int]]]:
        """Return each face as a 2×2 grid of colour indices.

        Keys: 'U', 'D', 'F', 'B', 'L', 'R'.
        Values: [[top-left, top-right], [bottom-left, bottom-right]].
        """
        faces: dict[str, list[list[int]]] = {}
        for idx, name in enumerate(FACE_NAMES):
            start = idx * STICKERS_PER_FACE
            faces[name] = [
                self.state[start : start + 2],
                self.state[start + 2 : start + 4],
            ]
        return faces

    def sticker_score(self) -> float:
        """Fraction of stickers in their solved position (0.0–1.0)."""
        return sum(
            1 for i in range(NUM_STICKERS) if self.state[i] == SOLVED_STATE[i]
        ) / NUM_STICKERS

    # ── Utilities ─────────────────────────────────────────────────────────

    def clone(self) -> PocketCube:
        """Create an independent deep copy."""
        new = PocketCube()
        new.state = list(self.state)
        new._steps = self._steps
        new._max_steps = self._max_steps
        new._scramble_depth = self._scramble_depth
        new._scramble_moves = list(self._scramble_moves)
        return new

    @staticmethod
    def solution_for_scramble(scramble_moves: list[int]) -> list[int]:
        """Return the move sequence that undoes *scramble_moves*.

        Simply reverses the sequence and replaces each move with its inverse.
        """
        return [INVERSE_ACTION[m] for m in reversed(scramble_moves)]

    @staticmethod
    def action_name(action: int) -> str:
        """Human-readable name for an action index."""
        return ACTION_NAMES[action]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PocketCube):
            return NotImplemented
        return self.state == other.state

    def __repr__(self) -> str:
        return f"PocketCube(solved={self.is_solved()}, steps={self._steps})"
