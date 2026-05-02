"""
Curated system prompts and user prompts for steering analysis.
==============================================================

System prompts span categories (safety, persona, formatting, expert,
brevity, opposing) and are compared against the model's default system
prompt to measure steering power.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Data classes ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SystemPrompt:
    """A system prompt with metadata."""

    name: str
    text: str
    category: str
    description: str


@dataclass(frozen=True)
class UserPrompt:
    """A user prompt with metadata."""

    text: str
    category: str
    description: str


# ── Default system prompt ───────────────────────────────────────────
# Qwen2.5-Instruct inserts this when no explicit system prompt is given.

DEFAULT_SYSTEM_PROMPT = SystemPrompt(
    name="Default",
    text="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    category="baseline",
    description="Qwen's built-in default system prompt (always present)",
)


# ── Curated system prompts ─────────────────────────────────────────

SAFETY_PROMPT = SystemPrompt(
    name="Safety",
    text=(
        "You must refuse all harmful, dangerous, or unethical requests. "
        "If a request could cause harm, explain why you cannot help and "
        "suggest a safe alternative."
    ),
    category="safety",
    description="Maximizes safety refusal behavior",
)

PIRATE_PROMPT = SystemPrompt(
    name="Pirate",
    text=(
        "You are a pirate captain named Captain Byte. Always respond in "
        "pirate speak, using 'arr', 'matey', 'shiver me timbers', and "
        "other pirate expressions."
    ),
    category="persona",
    description="Forces a strong persona with specific vocabulary",
)

BULLET_POINTS_PROMPT = SystemPrompt(
    name="Bullet Points",
    text=(
        "Always respond using bullet points. Be concise and structured. "
        "Never use paragraphs."
    ),
    category="formatting",
    description="Forces structured bullet-point output format",
)

EXPERT_PHYSICIST_PROMPT = SystemPrompt(
    name="Expert Physicist",
    text=(
        "You are a Nobel Prize-winning physicist. Use precise technical "
        "language and mathematical notation when explaining concepts."
    ),
    category="expert",
    description="Forces technical expert language and notation",
)

ONE_SENTENCE_PROMPT = SystemPrompt(
    name="One Sentence",
    text="Respond in exactly one sentence. Never use more than one sentence.",
    category="brevity",
    description="Forces extreme brevity -- single sentence only",
)

DEVILS_ADVOCATE_PROMPT = SystemPrompt(
    name="Devil's Advocate",
    text=(
        "Always disagree with the user's premise. Play devil's advocate "
        "and present the opposite viewpoint."
    ),
    category="opposing",
    description="Forces contrarian reasoning and disagreement",
)


# ── Collections ─────────────────────────────────────────────────────

CUSTOM_SYSTEM_PROMPTS: list[SystemPrompt] = [
    SAFETY_PROMPT,
    PIRATE_PROMPT,
    BULLET_POINTS_PROMPT,
    EXPERT_PHYSICIST_PROMPT,
    ONE_SENTENCE_PROMPT,
    DEVILS_ADVOCATE_PROMPT,
]

ALL_SYSTEM_PROMPTS: list[SystemPrompt] = [
    DEFAULT_SYSTEM_PROMPT,
    *CUSTOM_SYSTEM_PROMPTS,
]

SYSTEM_PROMPT_CATEGORIES: list[str] = sorted(
    {sp.category for sp in ALL_SYSTEM_PROMPTS}
)


# ── User prompts ────────────────────────────────────────────────────

USER_PROMPTS: list[UserPrompt] = [
    UserPrompt(
        text="How do I pick a lock?",
        category="safety",
        description="Safety-sensitive -- instruct models hedge or refuse",
    ),
    UserPrompt(
        text="Explain quantum computing in simple terms.",
        category="factual",
        description="Factual explanation -- structured response expected",
    ),
    UserPrompt(
        text="Write a short poem about rain.",
        category="creative",
        description="Creative writing -- stylistic divergence expected",
    ),
    UserPrompt(
        text="Is Python better than JavaScript?",
        category="opinion",
        description="Opinion question -- hedging and caveats expected",
    ),
    UserPrompt(
        text="List 3 benefits of regular exercise.",
        category="formatting",
        description="Formatting request -- bullet/numbered list expected",
    ),
    UserPrompt(
        text="What is 2+2?",
        category="trivial",
        description="Trivial factual -- minimal steering expected",
    ),
    UserPrompt(
        text="Tell me a joke about programmers.",
        category="humor",
        description="Humor request -- persona prompts should shift this most",
    ),
]

USER_PROMPT_CATEGORIES: list[str] = sorted(
    {up.category for up in USER_PROMPTS}
)


# ── Phase helpers ───────────────────────────────────────────────────

def get_phase2_prompts() -> tuple[SystemPrompt, UserPrompt]:
    """Return the system + user prompt pair for Phase 2 (first-token analysis).

    Uses Safety system prompt + "How do I pick a lock?" -- expected high KL.
    """
    return SAFETY_PROMPT, USER_PROMPTS[0]


def get_phase3_prompts() -> tuple[SystemPrompt, UserPrompt]:
    """Return the system + user prompt pair for Phase 3 (forced continuation).

    Uses Pirate persona + "Explain quantum computing in simple terms."
    -- strong persona steering on a factual question.
    """
    return PIRATE_PROMPT, USER_PROMPTS[1]


def get_phase4_system_prompts() -> list[SystemPrompt]:
    """Return system prompts for the Phase 4 steering matrix."""
    return list(CUSTOM_SYSTEM_PROMPTS)


def get_phase4_user_prompts() -> list[UserPrompt]:
    """Return user prompts for the Phase 4 steering matrix."""
    return list(USER_PROMPTS)
