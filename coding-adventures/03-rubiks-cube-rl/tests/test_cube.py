"""Tests for cube.py — 2x2 Pocket Cube environment.

Covers:
    - Solved state detection
    - All 6 moves: correctness, identity after 4×, inverse property
    - Scramble mechanics (length, no same-face consecutive, determinism)
    - One-hot tensor encoding
    - Step interface (reward, termination)
    - Sticker score
    - Clone independence
    - DLB corner fixedness
    - Constants and edge cases
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from cube import (
    PocketCube, StepResult,
    SOLVED_STATE, MOVE_CYCLES, INVERSE_ACTION, ACTION_NAMES, FACE_NAMES,
    FACE_COLORS, DLB_INDICES, NUM_STICKERS, NUM_ACTIONS, NUM_COLORS,
    NUM_FACES, STICKERS_PER_FACE, STATE_DIM, GODS_NUMBER,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Solved State
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolvedState:
    def test_initial_state_is_solved(self):
        cube = PocketCube()
        assert cube.is_solved()

    def test_solved_state_values(self):
        """Each face should have 4 stickers of the same colour index."""
        for i in range(NUM_FACES):
            for j in range(STICKERS_PER_FACE):
                assert SOLVED_STATE[i * STICKERS_PER_FACE + j] == i

    def test_is_solved_after_reset(self):
        cube = PocketCube()
        cube.scramble(5, rng=random.Random(42))
        cube.reset()
        assert cube.is_solved()

    def test_reset_returns_solved_state(self):
        cube = PocketCube()
        state = cube.reset()
        assert state == SOLVED_STATE


# ═══════════════════════════════════════════════════════════════════════════════
# Move Mechanics
# ═══════════════════════════════════════════════════════════════════════════════

class TestMoves:
    @pytest.mark.parametrize("action", range(NUM_ACTIONS))
    def test_move_changes_state(self, action):
        """Every single move should change the solved state."""
        cube = PocketCube()
        cube.apply_move(action)
        assert cube.state != SOLVED_STATE

    @pytest.mark.parametrize("action", range(NUM_ACTIONS))
    def test_move_four_times_identity(self, action):
        """Applying any quarter-turn 4 times returns to original state."""
        cube = PocketCube()
        for _ in range(4):
            cube.apply_move(action)
        assert cube.state == SOLVED_STATE

    @pytest.mark.parametrize("action", range(NUM_ACTIONS))
    def test_move_then_inverse_identity(self, action):
        """move + inverse = identity."""
        cube = PocketCube()
        cube.apply_move(action)
        cube.apply_move(INVERSE_ACTION[action])
        assert cube.state == SOLVED_STATE

    @pytest.mark.parametrize("action", range(NUM_ACTIONS))
    def test_inverse_then_move_identity(self, action):
        """inverse + move = identity."""
        cube = PocketCube()
        inv = INVERSE_ACTION[action]
        cube.apply_move(inv)
        cube.apply_move(action)
        assert cube.state == SOLVED_STATE

    @pytest.mark.parametrize("action", range(NUM_ACTIONS))
    def test_move_preserves_sticker_count(self, action):
        """Each colour should still appear exactly 4 times after any move."""
        cube = PocketCube()
        cube.apply_move(action)
        for color in range(NUM_COLORS):
            assert cube.state.count(color) == STICKERS_PER_FACE

    def test_all_moves_produce_different_states(self):
        """All 6 moves from solved should yield 6 distinct states."""
        states = set()
        for action in range(NUM_ACTIONS):
            cube = PocketCube()
            cube.apply_move(action)
            states.add(tuple(cube.state))
        assert len(states) == NUM_ACTIONS

    def test_u_twice_not_solved(self):
        """U applied twice (= U2) should not be solved."""
        cube = PocketCube()
        cube.apply_move(0)
        cube.apply_move(0)
        assert not cube.is_solved()

    def test_invalid_action_raises(self):
        cube = PocketCube()
        with pytest.raises(ValueError):
            cube.apply_move(-1)
        with pytest.raises(ValueError):
            cube.apply_move(6)

    def test_known_scramble_solution(self):
        """Scramble [U, R, F] should be solved by [F', R', U']."""
        cube = PocketCube()
        scramble = [0, 2, 4]  # U, R, F
        for m in scramble:
            cube.apply_move(m)
        assert not cube.is_solved()
        solution = PocketCube.solution_for_scramble(scramble)
        assert solution == [5, 3, 1]  # F', R', U'
        for m in solution:
            cube.apply_move(m)
        assert cube.is_solved()


# ═══════════════════════════════════════════════════════════════════════════════
# DLB Corner Fixedness
# ═══════════════════════════════════════════════════════════════════════════════

class TestDLBFixed:
    @pytest.mark.parametrize("action", range(NUM_ACTIONS))
    def test_dlb_stickers_fixed_single_move(self, action):
        """DLB corner stickers (6, 18, 15) must not move under any move."""
        cube = PocketCube()
        original = [SOLVED_STATE[i] for i in DLB_INDICES]
        cube.apply_move(action)
        for idx, orig_val in zip(DLB_INDICES, original):
            assert cube.state[idx] == orig_val

    def test_dlb_stickers_fixed_after_long_scramble(self):
        """DLB stickers stay fixed after many random moves."""
        cube = PocketCube()
        original = [SOLVED_STATE[i] for i in DLB_INDICES]
        cube.scramble(50, rng=random.Random(123))
        for idx, orig_val in zip(DLB_INDICES, original):
            assert cube.state[idx] == orig_val


# ═══════════════════════════════════════════════════════════════════════════════
# Scramble
# ═══════════════════════════════════════════════════════════════════════════════

class TestScramble:
    def test_scramble_1_not_solved(self):
        cube = PocketCube()
        cube.scramble(1, rng=random.Random(0))
        assert not cube.is_solved()

    def test_scramble_returns_correct_length(self):
        cube = PocketCube()
        moves = cube.scramble(5, rng=random.Random(42))
        assert len(moves) == 5

    def test_scramble_moves_are_valid(self):
        cube = PocketCube()
        moves = cube.scramble(10, rng=random.Random(42))
        for m in moves:
            assert 0 <= m < NUM_ACTIONS

    def test_scramble_no_same_face_consecutive(self):
        """Consecutive moves should never be on the same face."""
        cube = PocketCube()
        moves = cube.scramble(20, rng=random.Random(99))
        for i in range(1, len(moves)):
            assert moves[i] // 2 != moves[i - 1] // 2

    def test_scramble_deterministic_with_seed(self):
        """Same seed should produce same scramble."""
        c1, c2 = PocketCube(), PocketCube()
        m1 = c1.scramble(10, rng=random.Random(42))
        m2 = c2.scramble(10, rng=random.Random(42))
        assert m1 == m2
        assert c1.state == c2.state

    def test_scramble_different_seeds_differ(self):
        c1, c2 = PocketCube(), PocketCube()
        c1.scramble(10, rng=random.Random(1))
        c2.scramble(10, rng=random.Random(2))
        assert c1.state != c2.state

    def test_scramble_1_solvable_in_1_move(self):
        """A depth-1 scramble should be solvable by the inverse of the single move."""
        cube = PocketCube()
        moves = cube.scramble(1, rng=random.Random(42))
        cube.apply_move(INVERSE_ACTION[moves[0]])
        assert cube.is_solved()

    def test_scramble_sets_max_steps(self):
        cube = PocketCube()
        cube.scramble(5, rng=random.Random(42))
        assert cube._max_steps == 5 * 3 + 2

    def test_scramble_0_stays_solved(self):
        cube = PocketCube()
        moves = cube.scramble(0, rng=random.Random(42))
        assert moves == []
        assert cube.is_solved()

    def test_scramble_stores_moves(self):
        cube = PocketCube()
        moves = cube.scramble(3, rng=random.Random(42))
        assert cube._scramble_moves == moves
        assert cube._scramble_depth == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class TestTensor:
    def test_tensor_shape(self):
        cube = PocketCube()
        t = cube.to_tensor()
        assert t.shape == (STATE_DIM,)

    def test_tensor_dtype(self):
        cube = PocketCube()
        t = cube.to_tensor()
        assert t.dtype == torch.float32

    def test_tensor_one_hot_sum(self):
        """Each sticker contributes exactly one 1.0, total = 24."""
        cube = PocketCube()
        t = cube.to_tensor()
        assert t.sum().item() == pytest.approx(NUM_STICKERS)

    def test_tensor_one_hot_per_sticker(self):
        """Each block of 6 values should have exactly one 1.0."""
        cube = PocketCube()
        cube.scramble(5, rng=random.Random(42))
        t = cube.to_tensor()
        for i in range(NUM_STICKERS):
            block = t[i * NUM_COLORS : (i + 1) * NUM_COLORS]
            assert block.sum().item() == pytest.approx(1.0)
            assert block.max().item() == pytest.approx(1.0)

    def test_solved_tensor_pattern(self):
        """For solved state, sticker i should activate index i//4 within its block."""
        cube = PocketCube()
        t = cube.to_tensor()
        for i in range(NUM_STICKERS):
            expected_color = i // STICKERS_PER_FACE
            assert t[i * NUM_COLORS + expected_color].item() == pytest.approx(1.0)

    def test_different_states_different_tensors(self):
        c1, c2 = PocketCube(), PocketCube()
        c2.scramble(3, rng=random.Random(42))
        assert not torch.equal(c1.to_tensor(), c2.to_tensor())


# ═══════════════════════════════════════════════════════════════════════════════
# Step Interface
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep:
    def test_step_returns_step_result(self):
        cube = PocketCube()
        cube.scramble(1, rng=random.Random(42))
        result = cube.step(0)
        assert isinstance(result, StepResult)

    def test_step_increments_counter(self):
        cube = PocketCube()
        cube.scramble(2, rng=random.Random(42))
        cube.step(0)
        assert cube._steps == 1
        cube.step(0)
        assert cube._steps == 2

    def test_step_solved_gives_positive_reward(self):
        """Applying the inverse of a 1-move scramble should give +1.0 reward."""
        cube = PocketCube()
        moves = cube.scramble(1, rng=random.Random(42))
        result = cube.step(INVERSE_ACTION[moves[0]])
        assert result.reward == pytest.approx(1.0)
        assert result.done is True
        assert result.info["solved"] is True

    def test_step_unsolved_gives_negative_reward(self):
        cube = PocketCube()
        cube.scramble(5, rng=random.Random(42))
        result = cube.step(0)
        assert result.reward == pytest.approx(-0.02)

    def test_step_max_steps_terminates(self):
        cube = PocketCube()
        cube.scramble(1, rng=random.Random(42))
        cube._max_steps = 2
        cube.step(0)  # step 1
        result = cube.step(0)  # step 2 = max
        assert result.done is True

    def test_step_state_is_copy(self):
        """Returned state should be independent of internal state."""
        cube = PocketCube()
        cube.scramble(2, rng=random.Random(42))
        result = cube.step(0)
        result.state[0] = -999
        assert cube.state[0] != -999


# ═══════════════════════════════════════════════════════════════════════════════
# Sticker Score
# ═══════════════════════════════════════════════════════════════════════════════

class TestStickerScore:
    def test_solved_score_is_one(self):
        cube = PocketCube()
        assert cube.sticker_score() == pytest.approx(1.0)

    def test_scrambled_score_less_than_one(self):
        cube = PocketCube()
        cube.scramble(5, rng=random.Random(42))
        assert cube.sticker_score() < 1.0

    def test_score_in_valid_range(self):
        cube = PocketCube()
        cube.scramble(10, rng=random.Random(42))
        score = cube.sticker_score()
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Clone
# ═══════════════════════════════════════════════════════════════════════════════

class TestClone:
    def test_clone_equal(self):
        cube = PocketCube()
        cube.scramble(5, rng=random.Random(42))
        clone = cube.clone()
        assert clone.state == cube.state
        assert clone._steps == cube._steps

    def test_clone_independent(self):
        cube = PocketCube()
        cube.scramble(5, rng=random.Random(42))
        clone = cube.clone()
        clone.apply_move(0)
        assert clone.state != cube.state


# ═══════════════════════════════════════════════════════════════════════════════
# Constants & Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_num_stickers(self):
        assert NUM_STICKERS == 24

    def test_state_dim(self):
        assert STATE_DIM == 144

    def test_num_actions(self):
        assert NUM_ACTIONS == 6

    def test_action_names_count(self):
        assert len(ACTION_NAMES) == NUM_ACTIONS

    def test_face_names_count(self):
        assert len(FACE_NAMES) == NUM_FACES

    def test_face_colors_count(self):
        assert len(FACE_COLORS) == NUM_FACES

    def test_inverse_action_is_involution(self):
        """Inverting twice should return the original action."""
        for a in range(NUM_ACTIONS):
            assert INVERSE_ACTION[INVERSE_ACTION[a]] == a

    def test_gods_number(self):
        assert GODS_NUMBER == 14

    def test_move_cycles_all_present(self):
        for a in range(NUM_ACTIONS):
            assert a in MOVE_CYCLES


class TestUtilities:
    def test_action_name(self):
        assert PocketCube.action_name(0) == "U"
        assert PocketCube.action_name(1) == "U'"
        assert PocketCube.action_name(4) == "F"

    def test_solution_for_scramble(self):
        scramble = [0, 2, 4]  # U, R, F
        solution = PocketCube.solution_for_scramble(scramble)
        assert solution == [5, 3, 1]  # F', R', U'

    def test_solution_empty_scramble(self):
        assert PocketCube.solution_for_scramble([]) == []

    def test_repr(self):
        cube = PocketCube()
        r = repr(cube)
        assert "solved=True" in r

    def test_eq_same(self):
        c1, c2 = PocketCube(), PocketCube()
        assert c1 == c2

    def test_eq_different(self):
        c1, c2 = PocketCube(), PocketCube()
        c2.apply_move(0)
        assert c1 != c2

    def test_render_faces_structure(self):
        cube = PocketCube()
        faces = cube.render_faces()
        assert set(faces.keys()) == set(FACE_NAMES)
        for name in FACE_NAMES:
            assert len(faces[name]) == 2
            assert len(faces[name][0]) == 2
            assert len(faces[name][1]) == 2

    def test_render_faces_solved_colors(self):
        cube = PocketCube()
        faces = cube.render_faces()
        for idx, name in enumerate(FACE_NAMES):
            for row in faces[name]:
                for val in row:
                    assert val == idx
