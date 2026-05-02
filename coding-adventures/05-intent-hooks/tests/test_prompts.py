"""Tests for prompts.py — curated intent classification dataset."""

import sys
from pathlib import Path

# Allow running from the tests/ directory or the adventure root.
_adventure_root = Path(__file__).resolve().parent.parent
if str(_adventure_root) not in sys.path:
    sys.path.insert(0, str(_adventure_root))

import pytest

from prompts import (
    ALL_PROMPTS,
    AMBIGUOUS_PROMPTS,
    BENIGN_PROMPTS,
    HARMFUL_PROMPTS,
    JAILBREAK_PROMPTS,
    TRAIN_PROMPTS,
    IntentPrompt,
    get_categories,
    get_dataset_summary,
    get_prompts_by_category,
    get_prompts_by_label,
    get_subcategories,
    train_test_split,
)


# ─── IntentPrompt dataclass ────────────────────────────────────────────────

class TestIntentPrompt:
    def test_creation(self):
        p = IntentPrompt("Hello", "benign", "factual", "test")
        assert p.text == "Hello"
        assert p.label == "benign"
        assert p.category == "factual"
        assert p.subcategory == "test"

    def test_frozen(self):
        p = IntentPrompt("Hello", "benign", "factual", "test")
        with pytest.raises(AttributeError):
            p.text = "Goodbye"  # type: ignore[misc]

    def test_binary_label_benign(self):
        p = IntentPrompt("Hello", "benign", "factual", "test")
        assert p.binary_label == 0

    def test_binary_label_harmful(self):
        p = IntentPrompt("Bad", "harmful", "violence", "test")
        assert p.binary_label == 1

    def test_binary_label_ambiguous_raises(self):
        p = IntentPrompt("Maybe", "ambiguous", "dual_use", "test")
        with pytest.raises(ValueError, match="ambiguous"):
            _ = p.binary_label

    def test_binary_label_jailbreak_raises(self):
        p = IntentPrompt("DAN", "jailbreak", "jailbreak", "test")
        with pytest.raises(ValueError, match="jailbreak"):
            _ = p.binary_label

    def test_is_train_eligible_benign(self):
        p = IntentPrompt("Hello", "benign", "factual", "test")
        assert p.is_train_eligible is True

    def test_is_train_eligible_harmful(self):
        p = IntentPrompt("Bad", "harmful", "violence", "test")
        assert p.is_train_eligible is True

    def test_is_train_eligible_ambiguous(self):
        p = IntentPrompt("Maybe", "ambiguous", "dual_use", "test")
        assert p.is_train_eligible is False

    def test_is_train_eligible_jailbreak(self):
        p = IntentPrompt("DAN", "jailbreak", "jailbreak", "test")
        assert p.is_train_eligible is False

    def test_equality(self):
        a = IntentPrompt("X", "benign", "factual", "test")
        b = IntentPrompt("X", "benign", "factual", "test")
        assert a == b

    def test_inequality(self):
        a = IntentPrompt("X", "benign", "factual", "test")
        b = IntentPrompt("Y", "harmful", "violence", "test")
        assert a != b

    def test_hashable(self):
        a = IntentPrompt("X", "benign", "factual", "test")
        s = {a}
        assert a in s


# ─── Dataset integrity ──────────────────────────────────────────────────────

class TestDatasetIntegrity:
    def test_benign_count(self):
        assert len(BENIGN_PROMPTS) == 40

    def test_harmful_count(self):
        assert len(HARMFUL_PROMPTS) == 40

    def test_ambiguous_count(self):
        assert len(AMBIGUOUS_PROMPTS) == 10

    def test_jailbreak_count(self):
        assert len(JAILBREAK_PROMPTS) == 10

    def test_all_prompts_count(self):
        assert len(ALL_PROMPTS) == 100

    def test_train_prompts_count(self):
        assert len(TRAIN_PROMPTS) == 80

    def test_all_benign_labelled_benign(self):
        assert all(p.label == "benign" for p in BENIGN_PROMPTS)

    def test_all_harmful_labelled_harmful(self):
        assert all(p.label == "harmful" for p in HARMFUL_PROMPTS)

    def test_all_ambiguous_labelled_ambiguous(self):
        assert all(p.label == "ambiguous" for p in AMBIGUOUS_PROMPTS)

    def test_all_jailbreak_labelled_jailbreak(self):
        assert all(p.label == "jailbreak" for p in JAILBREAK_PROMPTS)

    def test_no_empty_texts(self):
        for p in ALL_PROMPTS:
            assert len(p.text.strip()) > 0, f"Empty text: {p}"

    def test_no_duplicate_texts(self):
        texts = [p.text for p in ALL_PROMPTS]
        assert len(texts) == len(set(texts)), "Duplicate prompt texts found"

    def test_all_have_category(self):
        for p in ALL_PROMPTS:
            assert len(p.category.strip()) > 0

    def test_all_have_subcategory(self):
        for p in ALL_PROMPTS:
            assert len(p.subcategory.strip()) > 0

    def test_train_eligible_matches_benign_harmful(self):
        eligible = [p for p in ALL_PROMPTS if p.is_train_eligible]
        assert len(eligible) == len(BENIGN_PROMPTS) + len(HARMFUL_PROMPTS)

    def test_benign_subcategories_are_diverse(self):
        cats = {p.category for p in BENIGN_PROMPTS}
        assert len(cats) >= 4

    def test_harmful_subcategories_are_diverse(self):
        cats = {p.category for p in HARMFUL_PROMPTS}
        assert len(cats) >= 4


# ─── Helper functions ────────────────────────────────────────────────────────

class TestHelpers:
    def test_get_prompts_by_label_benign(self):
        results = get_prompts_by_label("benign")
        assert len(results) == 40
        assert all(p.label == "benign" for p in results)

    def test_get_prompts_by_label_harmful(self):
        results = get_prompts_by_label("harmful")
        assert len(results) == 40
        assert all(p.label == "harmful" for p in results)

    def test_get_prompts_by_label_ambiguous(self):
        results = get_prompts_by_label("ambiguous")
        assert len(results) == 10

    def test_get_prompts_by_label_jailbreak(self):
        results = get_prompts_by_label("jailbreak")
        assert len(results) == 10

    def test_get_prompts_by_category(self):
        results = get_prompts_by_category("factual")
        assert len(results) > 0
        assert all(p.category == "factual" for p in results)

    def test_get_prompts_by_category_empty(self):
        results = get_prompts_by_category("nonexistent")
        assert results == []

    def test_get_categories(self):
        cats = get_categories()
        assert isinstance(cats, list)
        assert len(cats) >= 8  # benign(5) + harmful(5) + dual_use + jailbreak - overlaps
        assert cats == sorted(cats)

    def test_get_subcategories(self):
        subs = get_subcategories("factual")
        assert isinstance(subs, list)
        assert len(subs) > 0
        assert subs == sorted(subs)

    def test_get_subcategories_empty(self):
        subs = get_subcategories("nonexistent")
        assert subs == []

    def test_get_dataset_summary(self):
        summary = get_dataset_summary()
        assert summary["benign"] == 40
        assert summary["harmful"] == 40
        assert summary["ambiguous"] == 10
        assert summary["jailbreak"] == 10

    def test_get_dataset_summary_total(self):
        summary = get_dataset_summary()
        assert sum(summary.values()) == 100


# ─── Train/test split ────────────────────────────────────────────────────────

class TestTrainTestSplit:
    def test_split_sizes(self):
        train, test = train_test_split(TRAIN_PROMPTS, test_fraction=0.25)
        assert len(train) + len(test) == len(TRAIN_PROMPTS)

    def test_split_no_overlap(self):
        train, test = train_test_split(TRAIN_PROMPTS, test_fraction=0.25)
        train_texts = {p.text for p in train}
        test_texts = {p.text for p in test}
        assert train_texts.isdisjoint(test_texts)

    def test_split_stratified_both_labels_in_train(self):
        train, _ = train_test_split(TRAIN_PROMPTS, test_fraction=0.25)
        labels = {p.label for p in train}
        assert "benign" in labels
        assert "harmful" in labels

    def test_split_stratified_both_labels_in_test(self):
        _, test = train_test_split(TRAIN_PROMPTS, test_fraction=0.25)
        labels = {p.label for p in test}
        assert "benign" in labels
        assert "harmful" in labels

    def test_split_deterministic(self):
        train1, test1 = train_test_split(TRAIN_PROMPTS, seed=42)
        train2, test2 = train_test_split(TRAIN_PROMPTS, seed=42)
        assert [p.text for p in train1] == [p.text for p in train2]
        assert [p.text for p in test1] == [p.text for p in test2]

    def test_split_different_seeds(self):
        train1, _ = train_test_split(TRAIN_PROMPTS, seed=42)
        train2, _ = train_test_split(TRAIN_PROMPTS, seed=99)
        assert [p.text for p in train1] != [p.text for p in train2]

    def test_split_test_fraction_zero_point_five(self):
        train, test = train_test_split(TRAIN_PROMPTS, test_fraction=0.5)
        # Roughly equal
        assert abs(len(train) - len(test)) <= 2

    def test_split_all_are_train_eligible(self):
        train, test = train_test_split(TRAIN_PROMPTS)
        for p in train + test:
            assert p.is_train_eligible
