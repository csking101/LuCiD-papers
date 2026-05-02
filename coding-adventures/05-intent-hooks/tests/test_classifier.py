"""Tests for classifier.py — probing classifier for intent classification."""

import sys
from pathlib import Path

_adventure_root = Path(__file__).resolve().parent.parent
if str(_adventure_root) not in sys.path:
    sys.path.insert(0, str(_adventure_root))

import pytest
import torch

from classifier import (
    ConfusionMatrix,
    IntentProbe,
    LayerSweepResult,
    ProbeResult,
    compute_confusion_matrix,
    evaluate_probe,
    layer_sweep,
    train_probe,
)


# ─── ProbeResult ──────────────────────────────────────────────────────────────

class TestProbeResult:
    def test_creation(self):
        r = ProbeResult(layer=5, accuracy=0.9, precision=0.85, recall=0.95,
                        f1=0.9, train_loss=0.1, num_train=60, num_test=20)
        assert r.layer == 5
        assert r.accuracy == 0.9
        assert r.num_train == 60

    def test_frozen(self):
        r = ProbeResult(layer=0, accuracy=0.5, precision=0.5, recall=0.5,
                        f1=0.5, train_loss=0.5, num_train=10, num_test=5)
        with pytest.raises(AttributeError):
            r.accuracy = 0.99  # type: ignore[misc]


# ─── LayerSweepResult ────────────────────────────────────────────────────────

class TestLayerSweepResult:
    def _make_result(self, layer: int, accuracy: float) -> ProbeResult:
        return ProbeResult(layer=layer, accuracy=accuracy, precision=accuracy,
                           recall=accuracy, f1=accuracy, train_loss=0.1,
                           num_train=60, num_test=20)

    def test_empty(self):
        sweep = LayerSweepResult()
        assert sweep.results == []

    def test_best_layer(self):
        sweep = LayerSweepResult(results=[
            self._make_result(0, 0.6),
            self._make_result(1, 0.9),
            self._make_result(2, 0.8),
        ])
        assert sweep.best_layer == 1

    def test_best_accuracy(self):
        sweep = LayerSweepResult(results=[
            self._make_result(0, 0.6),
            self._make_result(1, 0.9),
        ])
        assert sweep.best_accuracy == 0.9

    def test_accuracies_in_order(self):
        sweep = LayerSweepResult(results=[
            self._make_result(2, 0.8),
            self._make_result(0, 0.6),
            self._make_result(1, 0.9),
        ])
        assert sweep.accuracies == [0.6, 0.9, 0.8]

    def test_layer_indices(self):
        sweep = LayerSweepResult(results=[
            self._make_result(2, 0.8),
            self._make_result(0, 0.6),
        ])
        assert sweep.layer_indices == [0, 2]

    def test_get_result(self):
        sweep = LayerSweepResult(results=[self._make_result(5, 0.75)])
        r = sweep.get_result(5)
        assert r.accuracy == 0.75

    def test_get_result_missing_raises(self):
        sweep = LayerSweepResult(results=[self._make_result(0, 0.5)])
        with pytest.raises(KeyError, match="layer 99"):
            sweep.get_result(99)

    def test_best_layer_empty_raises(self):
        sweep = LayerSweepResult()
        with pytest.raises(ValueError):
            _ = sweep.best_layer

    def test_best_accuracy_empty_raises(self):
        sweep = LayerSweepResult()
        with pytest.raises(ValueError):
            _ = sweep.best_accuracy


# ─── IntentProbe ──────────────────────────────────────────────────────────────

class TestIntentProbe:
    def test_creation(self):
        probe = IntentProbe(hidden_size=64)
        assert probe.hidden_size == 64

    def test_forward_shape(self):
        probe = IntentProbe(64)
        x = torch.randn(8, 64)
        out = probe(x)
        assert out.shape == (8, 1)

    def test_predict_shape(self):
        probe = IntentProbe(64)
        x = torch.randn(8, 64)
        preds = probe.predict(x)
        assert preds.shape == (8,)
        assert preds.dtype == torch.long

    def test_predict_binary(self):
        probe = IntentProbe(64)
        x = torch.randn(10, 64)
        preds = probe.predict(x)
        assert all(p in (0, 1) for p in preds.tolist())

    def test_predict_proba_range(self):
        probe = IntentProbe(64)
        x = torch.randn(10, 64)
        probs = probe.predict_proba(x)
        assert probs.shape == (10,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_threshold(self):
        probe = IntentProbe(64)
        x = torch.randn(10, 64)
        # With threshold 0.0, everything should be predicted as 1
        preds = probe.predict(x, threshold=0.0)
        assert (preds == 1).all()
        # With threshold 1.0, everything should be 0
        preds = probe.predict(x, threshold=1.0)
        assert (preds == 0).all()

    def test_gradients_flow(self):
        probe = IntentProbe(32)
        x = torch.randn(4, 32, requires_grad=True)
        out = probe(x)
        out.sum().backward()
        assert probe.linear.weight.grad is not None


# ─── Training ────────────────────────────────────────────────────────────────

class TestTrainProbe:
    def _make_linearly_separable_data(self, n=50, dim=16):
        """Create linearly separable data for reliable training tests."""
        torch.manual_seed(42)
        # Class 0: mean at -2, class 1: mean at +2
        x0 = torch.randn(n, dim) - 2.0
        x1 = torch.randn(n, dim) + 2.0
        features = torch.cat([x0, x1])
        labels = torch.cat([torch.zeros(n), torch.ones(n)])
        return features, labels

    def test_returns_probe_and_loss(self):
        features, labels = self._make_linearly_separable_data()
        probe, loss = train_probe(features, labels, epochs=10)
        assert isinstance(probe, IntentProbe)
        assert isinstance(loss, float)

    def test_loss_decreases(self):
        features, labels = self._make_linearly_separable_data()
        _, loss_10 = train_probe(features, labels, epochs=10)
        _, loss_200 = train_probe(features, labels, epochs=200)
        assert loss_200 < loss_10

    def test_probe_learns_separable_data(self):
        features, labels = self._make_linearly_separable_data(n=100, dim=32)
        probe, _ = train_probe(features, labels, epochs=200, lr=0.01)
        preds = probe.predict(features)
        accuracy = (preds == labels.long()).float().mean().item()
        assert accuracy > 0.9, f"Expected >90% accuracy, got {accuracy:.1%}"

    def test_infers_hidden_size(self):
        features = torch.randn(20, 48)
        labels = torch.zeros(20)
        probe, _ = train_probe(features, labels, epochs=5)
        assert probe.hidden_size == 48

    def test_custom_hidden_size(self):
        features = torch.randn(20, 48)
        labels = torch.zeros(20)
        probe, _ = train_probe(features, labels, hidden_size=48, epochs=5)
        assert probe.hidden_size == 48

    def test_probe_is_eval_mode_after_training(self):
        features = torch.randn(10, 16)
        labels = torch.zeros(10)
        probe, _ = train_probe(features, labels, epochs=5)
        assert not probe.training


# ─── Evaluation ──────────────────────────────────────────────────────────────

class TestEvaluateProbe:
    def test_perfect_predictions(self):
        probe = IntentProbe(16)
        # Force weights to produce correct predictions
        with torch.no_grad():
            probe.linear.weight.fill_(0)
            probe.linear.weight[0, 0] = 10.0  # first feature decides
            probe.linear.bias.fill_(-5.0)     # threshold at ~0.5 of first feature

        # Class 0: first feature = 0, Class 1: first feature = 1
        features = torch.zeros(4, 16)
        features[2, 0] = 1.0
        features[3, 0] = 1.0
        labels = torch.tensor([0, 0, 1, 1]).float()

        metrics = evaluate_probe(probe, features, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        probe = IntentProbe(8)
        with torch.no_grad():
            probe.linear.weight.fill_(0)
            probe.linear.weight[0, 0] = -10.0
            probe.linear.bias.fill_(5.0)  # Always predicts 1

        features = torch.zeros(4, 8)
        labels = torch.tensor([0, 0, 0, 0]).float()  # all benign

        metrics = evaluate_probe(probe, features, labels)
        assert metrics["accuracy"] == 0.0  # all wrong

    def test_returns_all_keys(self):
        probe = IntentProbe(8)
        features = torch.randn(10, 8)
        labels = torch.randint(0, 2, (10,)).float()
        metrics = evaluate_probe(probe, features, labels)
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_empty_data(self):
        probe = IntentProbe(8)
        features = torch.zeros(0, 8)
        labels = torch.zeros(0)
        metrics = evaluate_probe(probe, features, labels)
        assert metrics["accuracy"] == 0.0


# ─── Layer sweep ─────────────────────────────────────────────────────────────

class TestLayerSweep:
    def _make_sweep_data(self, num_layers=3, n=50, dim=16):
        """Create features for multiple layers with varying separability."""
        torch.manual_seed(42)
        train_features = {}
        test_features = {}

        for layer in range(num_layers):
            # Later layers have better separation
            sep = 0.5 + layer * 1.5
            x0_train = torch.randn(n, dim) - sep
            x1_train = torch.randn(n, dim) + sep
            x0_test = torch.randn(n // 4, dim) - sep
            x1_test = torch.randn(n // 4, dim) + sep

            train_features[layer] = torch.cat([x0_train, x1_train])
            test_features[layer] = torch.cat([x0_test, x1_test])

        train_labels = torch.cat([torch.zeros(n), torch.ones(n)])
        test_labels = torch.cat([torch.zeros(n // 4), torch.ones(n // 4)])

        return train_features, train_labels, test_features, test_labels

    def test_returns_results_and_probes(self):
        tf, tl, vf, vl = self._make_sweep_data()
        sweep, probes = layer_sweep(tf, tl, vf, vl, epochs=50)
        assert isinstance(sweep, LayerSweepResult)
        assert isinstance(probes, dict)
        assert len(sweep.results) == 3
        assert len(probes) == 3

    def test_later_layers_better(self):
        tf, tl, vf, vl = self._make_sweep_data(num_layers=3)
        sweep, _ = layer_sweep(tf, tl, vf, vl, epochs=100)
        accs = sweep.accuracies
        # Layer 2 (most separated) should have best accuracy
        assert accs[2] >= accs[0]

    def test_best_accuracy_is_high(self):
        tf, tl, vf, vl = self._make_sweep_data(num_layers=3)
        sweep, _ = layer_sweep(tf, tl, vf, vl, epochs=100)
        # Well-separated data should yield high accuracy at the best layer
        assert sweep.best_accuracy >= 0.9

    def test_probes_are_eval_mode(self):
        tf, tl, vf, vl = self._make_sweep_data()
        _, probes = layer_sweep(tf, tl, vf, vl, epochs=10)
        for probe in probes.values():
            assert not probe.training


# ─── ConfusionMatrix ──────────────────────────────────────────────────────────

class TestConfusionMatrix:
    def test_creation(self):
        cm = ConfusionMatrix(tp=10, fp=2, tn=8, fn=1)
        assert cm.tp == 10
        assert cm.fp == 2

    def test_accuracy(self):
        cm = ConfusionMatrix(tp=4, fp=1, tn=3, fn=2)
        assert cm.accuracy == pytest.approx(7 / 10)

    def test_precision(self):
        cm = ConfusionMatrix(tp=4, fp=1, tn=3, fn=2)
        assert cm.precision == pytest.approx(4 / 5)

    def test_recall(self):
        cm = ConfusionMatrix(tp=4, fp=1, tn=3, fn=2)
        assert cm.recall == pytest.approx(4 / 6)

    def test_f1(self):
        cm = ConfusionMatrix(tp=4, fp=1, tn=3, fn=2)
        p, r = 4/5, 4/6
        expected_f1 = 2 * p * r / (p + r)
        assert cm.f1 == pytest.approx(expected_f1)

    def test_all_zeros(self):
        cm = ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)
        assert cm.accuracy == 0.0
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.f1 == 0.0

    def test_perfect(self):
        cm = ConfusionMatrix(tp=5, fp=0, tn=5, fn=0)
        assert cm.accuracy == 1.0
        assert cm.precision == 1.0
        assert cm.recall == 1.0
        assert cm.f1 == 1.0

    def test_frozen(self):
        cm = ConfusionMatrix(tp=1, fp=0, tn=1, fn=0)
        with pytest.raises(AttributeError):
            cm.tp = 99  # type: ignore[misc]


class TestComputeConfusionMatrix:
    def test_returns_confusion_matrix(self):
        probe = IntentProbe(8)
        features = torch.randn(10, 8)
        labels = torch.randint(0, 2, (10,)).float()
        cm = compute_confusion_matrix(probe, features, labels)
        assert isinstance(cm, ConfusionMatrix)
        assert cm.tp + cm.fp + cm.tn + cm.fn == 10

    def test_consistent_with_evaluate(self):
        torch.manual_seed(42)
        probe = IntentProbe(16)
        features = torch.randn(20, 16)
        labels = torch.randint(0, 2, (20,)).float()

        cm = compute_confusion_matrix(probe, features, labels)
        metrics = evaluate_probe(probe, features, labels)

        assert cm.accuracy == pytest.approx(metrics["accuracy"], abs=1e-6)
