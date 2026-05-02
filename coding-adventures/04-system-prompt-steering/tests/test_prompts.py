"""Tests for prompts.py — curated system and user prompt collections."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts import (
    ALL_SYSTEM_PROMPTS,
    BULLET_POINTS_PROMPT,
    CUSTOM_SYSTEM_PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
    DEVILS_ADVOCATE_PROMPT,
    EXPERT_PHYSICIST_PROMPT,
    ONE_SENTENCE_PROMPT,
    PIRATE_PROMPT,
    SAFETY_PROMPT,
    SYSTEM_PROMPT_CATEGORIES,
    USER_PROMPT_CATEGORIES,
    USER_PROMPTS,
    SystemPrompt,
    UserPrompt,
    get_phase2_prompts,
    get_phase3_prompts,
    get_phase4_system_prompts,
    get_phase4_user_prompts,
)


# ── SystemPrompt dataclass ──────────────────────────────────────────

class TestSystemPromptDataclass:
    def test_fields(self):
        sp = SystemPrompt(name="Test", text="t", category="c", description="d")
        assert sp.name == "Test"
        assert sp.text == "t"
        assert sp.category == "c"
        assert sp.description == "d"

    def test_frozen(self):
        sp = SystemPrompt(name="Test", text="t", category="c", description="d")
        with pytest.raises(AttributeError):
            sp.name = "changed"

    def test_equality(self):
        a = SystemPrompt(name="A", text="t", category="c", description="d")
        b = SystemPrompt(name="A", text="t", category="c", description="d")
        assert a == b

    def test_inequality(self):
        a = SystemPrompt(name="A", text="t1", category="c", description="d")
        b = SystemPrompt(name="A", text="t2", category="c", description="d")
        assert a != b


# ── UserPrompt dataclass ────────────────────────────────────────────

class TestUserPromptDataclass:
    def test_fields(self):
        up = UserPrompt(text="Hello?", category="test", description="desc")
        assert up.text == "Hello?"
        assert up.category == "test"
        assert up.description == "desc"

    def test_frozen(self):
        up = UserPrompt(text="Hello?", category="test", description="desc")
        with pytest.raises(AttributeError):
            up.text = "changed"

    def test_equality(self):
        a = UserPrompt(text="X", category="c", description="d")
        b = UserPrompt(text="X", category="c", description="d")
        assert a == b


# ── Default system prompt ───────────────────────────────────────────

class TestDefaultSystemPrompt:
    def test_has_qwen_text(self):
        assert "Qwen" in DEFAULT_SYSTEM_PROMPT.text

    def test_category_is_baseline(self):
        assert DEFAULT_SYSTEM_PROMPT.category == "baseline"

    def test_name(self):
        assert DEFAULT_SYSTEM_PROMPT.name == "Default"


# ── System prompt collections ──────────────────────────────────────

class TestSystemPromptCollections:
    def test_custom_count(self):
        assert len(CUSTOM_SYSTEM_PROMPTS) == 6

    def test_all_includes_default(self):
        assert DEFAULT_SYSTEM_PROMPT in ALL_SYSTEM_PROMPTS

    def test_all_count(self):
        assert len(ALL_SYSTEM_PROMPTS) == 7

    def test_all_includes_all_custom(self):
        for sp in CUSTOM_SYSTEM_PROMPTS:
            assert sp in ALL_SYSTEM_PROMPTS

    def test_unique_names(self):
        names = [sp.name for sp in ALL_SYSTEM_PROMPTS]
        assert len(names) == len(set(names))

    def test_unique_categories(self):
        # Each custom prompt has a different category
        cats = [sp.category for sp in CUSTOM_SYSTEM_PROMPTS]
        assert len(cats) == len(set(cats))

    def test_all_have_nonempty_text(self):
        for sp in ALL_SYSTEM_PROMPTS:
            assert len(sp.text) > 10

    def test_all_have_descriptions(self):
        for sp in ALL_SYSTEM_PROMPTS:
            assert len(sp.description) > 5

    def test_individual_prompts_in_custom(self):
        expected = {SAFETY_PROMPT, PIRATE_PROMPT, BULLET_POINTS_PROMPT,
                    EXPERT_PHYSICIST_PROMPT, ONE_SENTENCE_PROMPT, DEVILS_ADVOCATE_PROMPT}
        assert set(CUSTOM_SYSTEM_PROMPTS) == expected

    def test_categories_sorted(self):
        assert SYSTEM_PROMPT_CATEGORIES == sorted(SYSTEM_PROMPT_CATEGORIES)

    def test_categories_complete(self):
        all_cats = {sp.category for sp in ALL_SYSTEM_PROMPTS}
        assert set(SYSTEM_PROMPT_CATEGORIES) == all_cats


# ── User prompt collections ─────────────────────────────────────────

class TestUserPromptCollections:
    def test_count(self):
        assert len(USER_PROMPTS) == 7

    def test_unique_texts(self):
        texts = [up.text for up in USER_PROMPTS]
        assert len(texts) == len(set(texts))

    def test_unique_categories(self):
        cats = [up.category for up in USER_PROMPTS]
        assert len(cats) == len(set(cats))

    def test_all_have_nonempty_text(self):
        for up in USER_PROMPTS:
            assert len(up.text) > 5

    def test_all_have_descriptions(self):
        for up in USER_PROMPTS:
            assert len(up.description) > 5

    def test_categories_sorted(self):
        assert USER_PROMPT_CATEGORIES == sorted(USER_PROMPT_CATEGORIES)

    def test_categories_complete(self):
        all_cats = {up.category for up in USER_PROMPTS}
        assert set(USER_PROMPT_CATEGORIES) == all_cats

    def test_expected_categories_present(self):
        cats = {up.category for up in USER_PROMPTS}
        assert "safety" in cats
        assert "factual" in cats
        assert "creative" in cats
        assert "trivial" in cats


# ── Phase helpers ───────────────────────────────────────────────────

class TestPhaseHelpers:
    def test_phase2_returns_tuple(self):
        sys_p, user_p = get_phase2_prompts()
        assert isinstance(sys_p, SystemPrompt)
        assert isinstance(user_p, UserPrompt)

    def test_phase2_safety_focus(self):
        sys_p, user_p = get_phase2_prompts()
        assert sys_p.category == "safety"
        assert "lock" in user_p.text.lower()

    def test_phase3_returns_tuple(self):
        sys_p, user_p = get_phase3_prompts()
        assert isinstance(sys_p, SystemPrompt)
        assert isinstance(user_p, UserPrompt)

    def test_phase3_persona_focus(self):
        sys_p, user_p = get_phase3_prompts()
        assert sys_p.category == "persona"
        assert "quantum" in user_p.text.lower()

    def test_phase4_system_prompts_are_custom(self):
        prompts = get_phase4_system_prompts()
        assert len(prompts) == len(CUSTOM_SYSTEM_PROMPTS)
        assert DEFAULT_SYSTEM_PROMPT not in prompts

    def test_phase4_user_prompts(self):
        prompts = get_phase4_user_prompts()
        assert len(prompts) == len(USER_PROMPTS)

    def test_phase4_returns_copies(self):
        # Returns new lists, not the originals
        sys_p = get_phase4_system_prompts()
        sys_p.append(DEFAULT_SYSTEM_PROMPT)
        assert len(get_phase4_system_prompts()) == len(CUSTOM_SYSTEM_PROMPTS)

        user_p = get_phase4_user_prompts()
        user_p.pop()
        assert len(get_phase4_user_prompts()) == len(USER_PROMPTS)
