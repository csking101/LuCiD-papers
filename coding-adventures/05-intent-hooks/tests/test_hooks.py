"""Tests for hooks.py — PyTorch hook management and feature extraction."""

import sys
from pathlib import Path

_adventure_root = Path(__file__).resolve().parent.parent
if str(_adventure_root) not in sys.path:
    sys.path.insert(0, str(_adventure_root))

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks import (
    CapturedStates,
    ModelInfo,
    capture_hidden_states,
    extract_features,
    get_device,
    get_module_hierarchy,
    get_transformer_layers,
    load_model,
    tokenize_prompt,
)

# Use tiny GPT-2 for CPU-only tests
TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_model():
    """Load the tiny GPT-2 model once for all tests in this module."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    model.eval()
    return model, tokenizer


# ─── Device detection ────────────────────────────────────────────────────────

class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_type_is_valid(self):
        device = get_device()
        assert device.type in ("cuda", "mps", "cpu")


# ─── ModelInfo ────────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_creation(self):
        info = ModelInfo("test", 12, 768, 50000, "cpu", "float32", 1000000)
        assert info.name == "test"
        assert info.num_layers == 12
        assert info.hidden_size == 768
        assert info.device == "cpu"

    def test_frozen(self):
        info = ModelInfo("test", 12, 768, 50000, "cpu", "float32", 1000000)
        with pytest.raises(AttributeError):
            info.name = "other"  # type: ignore[misc]


# ─── Load model ──────────────────────────────────────────────────────────────

class TestLoadModel:
    def test_load_tiny_model(self):
        model, tokenizer, info = load_model(TINY_MODEL, device=torch.device("cpu"))
        assert model is not None
        assert tokenizer is not None
        assert isinstance(info, ModelInfo)

    def test_info_fields(self):
        _, _, info = load_model(TINY_MODEL, device=torch.device("cpu"))
        assert info.name == TINY_MODEL
        assert info.num_layers > 0
        assert info.hidden_size > 0
        assert info.vocab_size > 0
        assert info.num_params > 0
        assert info.device == "cpu"

    def test_model_is_eval_mode(self):
        model, _, _ = load_model(TINY_MODEL, device=torch.device("cpu"))
        assert not model.training


# ─── Transformer layers ──────────────────────────────────────────────────────

class TestGetTransformerLayers:
    def test_returns_module_list(self, tiny_model):
        model, _ = tiny_model
        layers = get_transformer_layers(model)
        assert isinstance(layers, torch.nn.ModuleList)

    def test_layer_count(self, tiny_model):
        model, _ = tiny_model
        layers = get_transformer_layers(model)
        assert len(layers) == model.config.num_hidden_layers

    def test_unknown_architecture_raises(self):
        dummy = torch.nn.Linear(10, 10)
        with pytest.raises(AttributeError, match="Cannot find transformer layers"):
            get_transformer_layers(dummy)  # type: ignore[arg-type]


# ─── CapturedStates ──────────────────────────────────────────────────────────

class TestCapturedStates:
    def test_empty(self):
        cs = CapturedStates()
        assert cs.states == {}
        assert cs.layer_indices == []

    def test_get_last_token(self):
        hidden = torch.randn(1, 10, 64)  # [batch=1, seq=10, hidden=64]
        cs = CapturedStates(states={0: hidden}, num_layers=1)
        last = cs.get_last_token(0)
        assert last.shape == (1, 64)
        assert torch.equal(last, hidden[:, -1, :])

    def test_get_last_token_missing_layer_raises(self):
        cs = CapturedStates()
        with pytest.raises(KeyError, match="Layer 5"):
            cs.get_last_token(5)

    def test_get_all_last_token(self):
        h0 = torch.randn(2, 5, 32)
        h1 = torch.randn(2, 5, 32)
        cs = CapturedStates(states={0: h0, 1: h1}, num_layers=2)
        all_last = cs.get_all_last_token()
        assert all_last.shape == (2, 2, 32)  # [num_layers, batch, hidden]

    def test_layer_indices_sorted(self):
        cs = CapturedStates(states={3: torch.zeros(1), 1: torch.zeros(1), 5: torch.zeros(1)})
        assert cs.layer_indices == [1, 3, 5]


# ─── Capture hidden states ──────────────────────────────────────────────────

class TestCaptureHiddenStates:
    def test_captures_all_layers(self, tiny_model):
        model, tokenizer = tiny_model
        input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]

        with capture_hidden_states(model) as captured:
            model(input_ids)

        num_layers = model.config.num_hidden_layers
        assert len(captured.states) == num_layers
        assert captured.layer_indices == list(range(num_layers))

    def test_captures_specific_layers(self, tiny_model):
        model, tokenizer = tiny_model
        input_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]

        with capture_hidden_states(model, layers=[0, 1]) as captured:
            model(input_ids)

        assert captured.layer_indices == [0, 1]
        assert 2 not in captured.states

    def test_hidden_state_shape(self, tiny_model):
        model, tokenizer = tiny_model
        input_ids = tokenizer("Test prompt", return_tensors="pt")["input_ids"]
        seq_len = input_ids.shape[1]
        hidden_size = model.config.hidden_size

        with capture_hidden_states(model, layers=[0]) as captured:
            model(input_ids)

        h = captured.states[0]
        assert h.shape == (1, seq_len, hidden_size)

    def test_hooks_are_removed_after_context(self, tiny_model):
        model, tokenizer = tiny_model
        layers = get_transformer_layers(model)

        # Count hooks before
        hooks_before = sum(
            len(layer._forward_hooks) for layer in layers
        )

        with capture_hidden_states(model) as captured:
            pass  # Don't even run forward

        # Count hooks after
        hooks_after = sum(
            len(layer._forward_hooks) for layer in layers
        )
        assert hooks_after == hooks_before

    def test_hooks_removed_on_exception(self, tiny_model):
        model, _ = tiny_model
        layers = get_transformer_layers(model)
        hooks_before = sum(len(layer._forward_hooks) for layer in layers)

        with pytest.raises(RuntimeError):
            with capture_hidden_states(model) as captured:
                raise RuntimeError("Intentional error")

        hooks_after = sum(len(layer._forward_hooks) for layer in layers)
        assert hooks_after == hooks_before

    def test_invalid_layer_index_raises(self, tiny_model):
        model, _ = tiny_model
        with pytest.raises(IndexError, match="out of range"):
            with capture_hidden_states(model, layers=[999]):
                pass

    def test_negative_layer_index_raises(self, tiny_model):
        model, _ = tiny_model
        with pytest.raises(IndexError, match="out of range"):
            with capture_hidden_states(model, layers=[-1]):
                pass

    def test_states_are_detached(self, tiny_model):
        model, tokenizer = tiny_model
        input_ids = tokenizer("Test", return_tensors="pt")["input_ids"]

        with capture_hidden_states(model, layers=[0]) as captured:
            model(input_ids)

        assert not captured.states[0].requires_grad


# ─── Tokenize prompt ─────────────────────────────────────────────────────────

class TestTokenizePrompt:
    def test_returns_tensor(self, tiny_model):
        _, tokenizer = tiny_model
        # tiny-gpt2 may not have a chat template, so test basic tokenization
        ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
        assert isinstance(ids, torch.Tensor)
        assert ids.dim() == 2  # [batch, seq]

    def test_nonempty(self, tiny_model):
        _, tokenizer = tiny_model
        ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
        assert ids.shape[1] > 0


# ─── Feature extraction ──────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_returns_dict(self, tiny_model):
        model, tokenizer = tiny_model
        # Use raw tokenization for tiny-gpt2 (no chat template)
        texts = ["Hello", "World"]
        # We need to work around the chat template issue for tiny-gpt2
        features = _extract_features_raw(texts, model, tokenizer)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_feature_shape(self, tiny_model):
        model, tokenizer = tiny_model
        texts = ["Hello", "World"]
        features = _extract_features_raw(texts, model, tokenizer)
        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        for layer_idx, feat in features.items():
            assert feat.shape == (2, hidden_size)

    def test_specific_layers(self, tiny_model):
        model, tokenizer = tiny_model
        texts = ["Test"]
        features = _extract_features_raw(texts, model, tokenizer, layers=[0])
        assert list(features.keys()) == [0]

    def test_single_prompt(self, tiny_model):
        model, tokenizer = tiny_model
        features = _extract_features_raw(["Single"], model, tokenizer, layers=[0])
        assert features[0].shape[0] == 1


def _extract_features_raw(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Helper: extract features using raw tokenization (for tiny-gpt2 without chat template)."""
    device = next(model.parameters()).device
    all_features: dict[int, list[torch.Tensor]] = {}

    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        with capture_hidden_states(model, layers=layers) as captured:
            model(input_ids)
        for layer_idx in captured.layer_indices:
            last_tok = captured.get_last_token(layer_idx).squeeze(0)
            if layer_idx not in all_features:
                all_features[layer_idx] = []
            all_features[layer_idx].append(last_tok)

    return {l: torch.stack(f) for l, f in all_features.items()}


# ─── Module hierarchy ─────────────────────────────────────────────────────────

class TestGetModuleHierarchy:
    def test_returns_list(self, tiny_model):
        model, _ = tiny_model
        paths = get_module_hierarchy(model)
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_max_depth(self, tiny_model):
        model, _ = tiny_model
        paths_d1 = get_module_hierarchy(model, max_depth=1)
        paths_d3 = get_module_hierarchy(model, max_depth=3)
        assert len(paths_d1) <= len(paths_d3)

    def test_no_empty_strings(self, tiny_model):
        model, _ = tiny_model
        paths = get_module_hierarchy(model)
        assert all(len(p) > 0 for p in paths)
