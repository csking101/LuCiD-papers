"""
Tests for the Preference Database.
====================================

Covers:
- Adding preferences and length tracking
- Label validation
- Preference pair properties
- Sampling (batch, empty DB, larger-than-available)
- Conversion to training data format
- Analytics: count_by_preference, preference_patterns, summary_stats
- Edge cases (empty DB, all ties, single entry)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import DOWN, RIGHT, Trajectory
from preferences import PreferenceDatabase, PreferencePair


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _make_trajectory(length: int, num_turns: int = 0) -> Trajectory:
    """Create a fake trajectory with given length and approximate turns."""
    states = [np.array([i * 0.1, 0.0], dtype=np.float32) for i in range(length)]
    positions = [(i, 0) for i in range(length + 1)]
    # Create actions with desired number of turns
    if length == 0:
        actions = []
    elif num_turns == 0:
        actions = [RIGHT] * length
    else:
        actions = []
        current_action = RIGHT
        steps_per_segment = max(length // (num_turns + 1), 1)
        for i in range(length):
            if i > 0 and i % steps_per_segment == 0 and len(set(actions)) < num_turns + 1:
                current_action = DOWN if current_action == RIGHT else RIGHT
            actions.append(current_action)

    return Trajectory(
        states=states,
        actions=actions,
        rewards=[0.0] * length,
        positions=positions,
        log_probs=[0.0] * length,
        values=[0.0] * length,
        done=True,
        reached_goal=False,
    )


# -----------------------------------------------------------------------
# PreferencePair
# -----------------------------------------------------------------------
class TestPreferencePair:
    def test_preferred_a(self):
        p = PreferencePair(
            traj_a=_make_trajectory(5),
            traj_b=_make_trajectory(10),
            label=0.0,
        )
        assert p.preferred == "A"

    def test_preferred_b(self):
        p = PreferencePair(
            traj_a=_make_trajectory(5),
            traj_b=_make_trajectory(10),
            label=1.0,
        )
        assert p.preferred == "B"

    def test_preferred_tie(self):
        p = PreferencePair(
            traj_a=_make_trajectory(5),
            traj_b=_make_trajectory(10),
            label=0.5,
        )
        assert p.preferred == "tie"

    def test_label_boundary_a(self):
        p = PreferencePair(
            traj_a=_make_trajectory(1),
            traj_b=_make_trajectory(1),
            label=0.24,
        )
        assert p.preferred == "A"

    def test_label_boundary_b(self):
        p = PreferencePair(
            traj_a=_make_trajectory(1),
            traj_b=_make_trajectory(1),
            label=0.76,
        )
        assert p.preferred == "B"


# -----------------------------------------------------------------------
# Database basics
# -----------------------------------------------------------------------
class TestDatabaseBasics:
    def test_empty(self):
        db = PreferenceDatabase()
        assert len(db) == 0

    def test_add_and_length(self):
        db = PreferenceDatabase()
        db.add(_make_trajectory(5), _make_trajectory(10), 0.0)
        assert len(db) == 1
        db.add(_make_trajectory(5), _make_trajectory(10), 1.0)
        assert len(db) == 2

    def test_getitem(self):
        db = PreferenceDatabase()
        ta = _make_trajectory(3)
        tb = _make_trajectory(7)
        db.add(ta, tb, 0.0)
        pair = db[0]
        assert pair.traj_a is ta
        assert pair.traj_b is tb
        assert pair.label == 0.0

    def test_invalid_label_low(self):
        db = PreferenceDatabase()
        with pytest.raises(AssertionError):
            db.add(_make_trajectory(1), _make_trajectory(1), -0.1)

    def test_invalid_label_high(self):
        db = PreferenceDatabase()
        with pytest.raises(AssertionError):
            db.add(_make_trajectory(1), _make_trajectory(1), 1.1)


# -----------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------
class TestSampling:
    def test_sample_batch(self):
        db = PreferenceDatabase()
        for _ in range(10):
            db.add(_make_trajectory(5), _make_trajectory(5), 0.0)
        batch = db.sample_batch(3)
        assert len(batch) == 3
        assert all(isinstance(p, PreferencePair) for p in batch)

    def test_sample_larger_than_available(self):
        db = PreferenceDatabase()
        db.add(_make_trajectory(5), _make_trajectory(5), 0.0)
        batch = db.sample_batch(10)
        assert len(batch) == 1

    def test_sample_empty_db(self):
        db = PreferenceDatabase()
        batch = db.sample_batch(5)
        assert batch == []

    def test_sample_no_replacement(self):
        db = PreferenceDatabase()
        for i in range(5):
            t = _make_trajectory(i + 1)
            db.add(t, t, 0.0)
        batch = db.sample_batch(5)
        # All 5 should be distinct objects
        ids = [id(p) for p in batch]
        assert len(set(ids)) == 5


# -----------------------------------------------------------------------
# Training data conversion
# -----------------------------------------------------------------------
class TestTrainingData:
    def test_format(self):
        db = PreferenceDatabase()
        ta = _make_trajectory(3)
        tb = _make_trajectory(5)
        db.add(ta, tb, 0.0)
        data = db.to_training_data()
        assert len(data) == 1
        sa, sb, label = data[0]
        assert sa.shape == (3, 2)
        assert sb.shape == (5, 2)
        assert label == 0.0

    def test_empty_db(self):
        db = PreferenceDatabase()
        data = db.to_training_data()
        assert data == []

    def test_dtype(self):
        db = PreferenceDatabase()
        db.add(_make_trajectory(4), _make_trajectory(4), 1.0)
        sa, sb, label = db.to_training_data()[0]
        assert sa.dtype == np.float32
        assert sb.dtype == np.float32


# -----------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------
class TestAnalytics:
    def test_count_by_preference(self):
        db = PreferenceDatabase()
        db.add(_make_trajectory(5), _make_trajectory(10), 0.0)  # A
        db.add(_make_trajectory(5), _make_trajectory(10), 1.0)  # B
        db.add(_make_trajectory(5), _make_trajectory(10), 0.5)  # tie
        db.add(_make_trajectory(5), _make_trajectory(10), 0.0)  # A
        counts = db.count_by_preference()
        assert counts == {"A": 2, "B": 1, "tie": 1}

    def test_count_empty(self):
        db = PreferenceDatabase()
        counts = db.count_by_preference()
        assert counts == {"A": 0, "B": 0, "tie": 0}

    def test_preference_patterns_shorter(self):
        db = PreferenceDatabase()
        # Always prefer shorter path
        for _ in range(10):
            db.add(_make_trajectory(5), _make_trajectory(15), 0.0)
        patterns = db.preference_patterns()
        assert patterns["shorter_paths"] == "10/10"

    def test_preference_patterns_empty(self):
        db = PreferenceDatabase()
        patterns = db.preference_patterns()
        assert "status" in patterns

    def test_preference_patterns_all_ties(self):
        db = PreferenceDatabase()
        db.add(_make_trajectory(5), _make_trajectory(5), 0.5)
        patterns = db.preference_patterns()
        assert patterns["status"] == "All ties"

    def test_summary_stats_empty(self):
        db = PreferenceDatabase()
        stats = db.summary_stats()
        assert stats == {}

    def test_summary_stats_values(self):
        db = PreferenceDatabase()
        # Prefer short (3) over long (10)
        for _ in range(5):
            db.add(_make_trajectory(3), _make_trajectory(10), 0.0)
        stats = db.summary_stats()
        assert stats["avg_length_preferred"] == pytest.approx(3.0)
        assert stats["avg_length_other"] == pytest.approx(10.0)

    def test_summary_stats_with_ties_excluded(self):
        db = PreferenceDatabase()
        db.add(_make_trajectory(3), _make_trajectory(10), 0.0)
        db.add(_make_trajectory(5), _make_trajectory(5), 0.5)  # tie, excluded
        stats = db.summary_stats()
        assert stats["avg_length_preferred"] == pytest.approx(3.0)
        assert stats["avg_length_other"] == pytest.approx(10.0)


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------
class TestEdgeCases:
    def test_zero_length_trajectory(self):
        db = PreferenceDatabase()
        t0 = _make_trajectory(0)
        t1 = _make_trajectory(5)
        db.add(t0, t1, 1.0)
        data = db.to_training_data()
        assert data[0][0].shape == (0, 2)

    def test_many_preferences(self):
        db = PreferenceDatabase()
        for i in range(100):
            db.add(_make_trajectory(5), _make_trajectory(5), float(i % 2))
        assert len(db) == 100
        batch = db.sample_batch(20)
        assert len(batch) == 20
