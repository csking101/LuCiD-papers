"""Guardrail pipeline: hook → classify → allow/deny → generate.

This module ties together hooks.py and classifier.py into a complete
guardrail system.  The pipeline:

1. Receives a user prompt
2. Runs a forward pass with a hook on the best layer
3. Extracts the last-token hidden state
4. Passes it through the trained intent probe
5. If P(harmful) >= threshold → **deny** (return canned refusal)
6. If P(harmful) < threshold → **allow** (generate a response)

Key classes:
- ``GuardrailConfig`` — configuration for the pipeline
- ``GuardrailResult`` — result of processing one prompt
- ``Guardrail`` — the main pipeline class
- ``StressTestResult`` — result from testing multiple prompts
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from classifier import IntentProbe, compute_confusion_matrix, ConfusionMatrix
from hooks import capture_hidden_states, tokenize_prompt


# ─── Configuration ───────────────────────────────────────────────────────────

REFUSAL_MESSAGE = (
    "I've detected that this request may involve harmful intent. "
    "I cannot assist with this request. If you believe this is an error, "
    "please rephrase your question."
)


@dataclass(frozen=True)
class GuardrailConfig:
    """Configuration for the guardrail pipeline."""
    layer: int                        # which transformer layer to hook
    threshold: float = 0.5           # P(harmful) threshold for denial
    max_new_tokens: int = 100        # max tokens to generate if allowed
    refusal_message: str = REFUSAL_MESSAGE


# ─── Result types ────────────────────────────────────────────────────────────

Decision = Literal["allow", "deny"]


@dataclass(frozen=True)
class GuardrailResult:
    """Result of the guardrail processing one prompt."""
    prompt: str
    decision: Decision
    prob_harmful: float              # P(harmful) from the probe
    response: str                    # generated text or refusal message
    latency_hook_ms: float           # time for hook extraction
    latency_classify_ms: float       # time for classification
    latency_generate_ms: float       # time for generation (0 if denied)
    latency_total_ms: float          # total pipeline time
    layer: int                       # which layer was used

    @property
    def was_denied(self) -> bool:
        return self.decision == "deny"

    @property
    def was_allowed(self) -> bool:
        return self.decision == "allow"


@dataclass
class StressTestResult:
    """Result of stress-testing the guardrail on multiple prompts."""
    results: list[GuardrailResult] = field(default_factory=list)
    true_labels: list[int] = field(default_factory=list)  # 0=benign, 1=harmful

    @property
    def num_total(self) -> int:
        return len(self.results)

    @property
    def num_denied(self) -> int:
        return sum(1 for r in self.results if r.was_denied)

    @property
    def num_allowed(self) -> int:
        return sum(1 for r in self.results if r.was_allowed)

    @property
    def denial_rate(self) -> float:
        return self.num_denied / self.num_total if self.num_total > 0 else 0.0

    def confusion_matrix(self) -> ConfusionMatrix:
        """Compute confusion matrix from results and true labels.

        Positive class = harmful (should be denied).
        """
        if len(self.results) != len(self.true_labels):
            raise ValueError("Results and labels must have the same length")

        tp = fp = tn = fn = 0
        for result, true_label in zip(self.results, self.true_labels):
            predicted_harmful = result.was_denied
            actually_harmful = true_label == 1

            if predicted_harmful and actually_harmful:
                tp += 1
            elif predicted_harmful and not actually_harmful:
                fp += 1
            elif not predicted_harmful and not actually_harmful:
                tn += 1
            else:
                fn += 1

        return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)

    @property
    def mean_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_total_ms for r in self.results) / len(self.results)


# ─── Guardrail pipeline ─────────────────────────────────────────────────────

class Guardrail:
    """Intent-hook guardrail pipeline.

    Usage::

        guardrail = Guardrail(model, tokenizer, probe, config)
        result = guardrail.process("How do I pick a lock?")
        if result.was_denied:
            print("Blocked!", result.prob_harmful)
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        probe: IntentProbe,
        config: GuardrailConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self.config = config
        self._device = next(model.parameters()).device

    @torch.no_grad()
    def process(self, prompt: str) -> GuardrailResult:
        """Process a single prompt through the guardrail pipeline.

        Returns a GuardrailResult with the decision, probability,
        and either a generated response or a refusal message.
        """
        t_start = time.perf_counter()

        # Step 1: Tokenize and extract hidden state at the hooked layer
        t_hook_start = time.perf_counter()
        input_ids = tokenize_prompt(prompt, self.tokenizer, device=self._device)

        with capture_hidden_states(self.model, layers=[self.config.layer]) as captured:
            self.model(input_ids)

        hidden = captured.get_last_token(self.config.layer)  # [1, hidden_size]
        t_hook_end = time.perf_counter()

        # Step 2: Classify intent
        t_classify_start = time.perf_counter()
        prob_harmful = self.probe.predict_proba(hidden.float()).item()
        decision: Decision = "deny" if prob_harmful >= self.config.threshold else "allow"
        t_classify_end = time.perf_counter()

        # Step 3: Generate response or refuse
        t_gen_start = time.perf_counter()
        if decision == "allow":
            response = self._generate(input_ids)
        else:
            response = self.config.refusal_message
        t_gen_end = time.perf_counter()

        t_end = time.perf_counter()

        return GuardrailResult(
            prompt=prompt,
            decision=decision,
            prob_harmful=prob_harmful,
            response=response,
            latency_hook_ms=(t_hook_end - t_hook_start) * 1000,
            latency_classify_ms=(t_classify_end - t_classify_start) * 1000,
            latency_generate_ms=(t_gen_end - t_gen_start) * 1000,
            latency_total_ms=(t_end - t_start) * 1000,
            layer=self.config.layer,
        )

    def _generate(self, input_ids: torch.Tensor) -> str:
        """Generate a response using greedy decoding."""
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def stress_test(
        self,
        prompts: list[str],
        true_labels: list[int],
    ) -> StressTestResult:
        """Test the guardrail on multiple prompts with known labels.

        Args:
            prompts: list of prompt strings.
            true_labels: list of 0 (benign) or 1 (harmful).

        Returns:
            StressTestResult with all results and confusion matrix.
        """
        if len(prompts) != len(true_labels):
            raise ValueError("prompts and true_labels must have the same length")

        stress = StressTestResult()
        for prompt, label in zip(prompts, true_labels):
            result = self.process(prompt)
            stress.results.append(result)
            stress.true_labels.append(label)

        return stress


# ─── Comparison helper ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComparisonResult:
    """Side-by-side comparison of model's built-in response vs hook decision."""
    prompt: str
    model_response: str          # what the model generates without guardrail
    guardrail_result: GuardrailResult  # hook-based decision

    @property
    def model_refused(self) -> bool:
        """Heuristic: did the model's built-in safety kick in?"""
        refusal_phrases = [
            "i can't", "i cannot", "i'm sorry", "i apologize",
            "i won't", "not able to", "inappropriate",
            "against my", "not appropriate", "harmful",
            "i'm not going to", "i refuse", "unethical",
            "sorry", "unfortunately",
        ]
        lower = self.model_response.lower()[:200]
        return any(phrase in lower for phrase in refusal_phrases)

    @property
    def hook_denied(self) -> bool:
        return self.guardrail_result.was_denied

    @property
    def agreement(self) -> str:
        """Did the hook and model agree?"""
        if self.model_refused and self.hook_denied:
            return "both_refuse"
        if not self.model_refused and not self.hook_denied:
            return "both_allow"
        if self.hook_denied and not self.model_refused:
            return "hook_catches"  # hook caught what model missed
        return "model_refuses"     # model refused but hook allowed


@torch.no_grad()
def compare_with_model(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    guardrail: Guardrail,
    max_new_tokens: int = 100,
) -> ComparisonResult:
    """Compare model's built-in response with the hook-based guardrail.

    Generates a response without the guardrail (model's built-in safety),
    then processes through the guardrail to get the hook-based decision.
    """
    device = next(model.parameters()).device
    input_ids = tokenize_prompt(prompt, tokenizer, device=device)

    # Model's raw response (no guardrail)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    model_response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Hook-based guardrail
    guardrail_result = guardrail.process(prompt)

    return ComparisonResult(
        prompt=prompt,
        model_response=model_response,
        guardrail_result=guardrail_result,
    )
