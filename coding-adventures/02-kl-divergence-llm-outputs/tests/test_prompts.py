"""
Tests for prompts.py — curated prompt sets and utilities.
"""

from __future__ import annotations

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import (
    ALL_PROMPTS,
    CATEGORY_DESCRIPTIONS,
    FORMATTING_PROMPTS,
    HEDGING_PROMPTS,
    HELPFULNESS_PROMPTS,
    PROMPTS_BY_CATEGORY,
    SAFETY_PROMPTS,
    STYLE_PROMPTS,
    Prompt,
    get_phase2_prompt,
    get_phase3_prompts,
)


class TestPromptDataclass:
    def test_prompt_fields(self):
        p = Prompt(text="test", category="cat", description="desc")
        assert p.text == "test"
        assert p.category == "cat"
        assert p.description == "desc"

    def test_prompt_frozen(self):
        p = Prompt(text="test", category="cat", description="desc")
        with pytest.raises(AttributeError):
            p.text = "changed"


class TestPromptSets:
    def test_safety_not_empty(self):
        assert len(SAFETY_PROMPTS) >= 3

    def test_helpfulness_not_empty(self):
        assert len(HELPFULNESS_PROMPTS) >= 3

    def test_style_not_empty(self):
        assert len(STYLE_PROMPTS) >= 3

    def test_hedging_not_empty(self):
        assert len(HEDGING_PROMPTS) >= 3

    def test_formatting_not_empty(self):
        assert len(FORMATTING_PROMPTS) >= 3

    def test_all_prompts_count(self):
        expected = (
            len(SAFETY_PROMPTS) + len(HELPFULNESS_PROMPTS)
            + len(STYLE_PROMPTS) + len(HEDGING_PROMPTS)
            + len(FORMATTING_PROMPTS)
        )
        assert len(ALL_PROMPTS) == expected

    def test_all_prompts_have_text(self):
        for p in ALL_PROMPTS:
            assert len(p.text) > 0
            assert len(p.category) > 0
            assert len(p.description) > 0

    def test_categories_match(self):
        for p in SAFETY_PROMPTS:
            assert p.category == "safety"
        for p in HELPFULNESS_PROMPTS:
            assert p.category == "helpfulness"
        for p in STYLE_PROMPTS:
            assert p.category == "style"
        for p in HEDGING_PROMPTS:
            assert p.category == "hedging"
        for p in FORMATTING_PROMPTS:
            assert p.category == "formatting"


class TestConvenienceGroupings:
    def test_prompts_by_category_keys(self):
        assert set(PROMPTS_BY_CATEGORY.keys()) == {
            "safety", "helpfulness", "style", "hedging", "formatting",
        }

    def test_prompts_by_category_values_are_strings(self):
        for cat, texts in PROMPTS_BY_CATEGORY.items():
            for t in texts:
                assert isinstance(t, str)
                assert len(t) > 0

    def test_category_descriptions_complete(self):
        for cat in PROMPTS_BY_CATEGORY:
            assert cat in CATEGORY_DESCRIPTIONS
            assert len(CATEGORY_DESCRIPTIONS[cat]) > 0


class TestPhaseHelpers:
    def test_get_phase2_prompt(self):
        p = get_phase2_prompt()
        assert isinstance(p, Prompt)
        assert len(p.text) > 0

    def test_get_phase3_prompts(self):
        prompts = get_phase3_prompts()
        assert len(prompts) == 5
        categories = {p.category for p in prompts}
        assert categories == {"safety", "helpfulness", "style", "hedging", "formatting"}

    def test_phase3_all_unique(self):
        prompts = get_phase3_prompts()
        texts = [p.text for p in prompts]
        assert len(texts) == len(set(texts))
