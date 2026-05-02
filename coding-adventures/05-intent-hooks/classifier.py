"""Probing classifier for intent classification on hidden states.

Pure PyTorch implementation — no sklearn dependency.  Everything stays on GPU.

The ``IntentProbe`` is a single ``nn.Linear(hidden_size, 1)`` trained with
``BCEWithLogitsLoss``.  This is a *linear probe*: if it achieves high
accuracy, it means the safety-relevant information is **linearly encoded**
in the hidden state at that layer.

Key classes:
- ``IntentProbe`` — single-layer binary classifier (benign=0, harmful=1)
- ``ProbeResult`` — per-layer accuracy, loss, and metadata
- ``LayerSweepResult`` — results across all layers

Key functions:
- ``train_probe()`` — train a probe on features from one layer
- ``evaluate_probe()`` — compute accuracy, precision, recall, F1
- ``layer_sweep()`` — train + evaluate probes at every layer
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbeResult:
    """Result of training and evaluating a probe at one layer."""
    layer: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_loss: float
    num_train: int
    num_test: int


@dataclass
class LayerSweepResult:
    """Results from probing all layers."""
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def best_layer(self) -> int:
        """Layer index with the highest accuracy."""
        if not self.results:
            raise ValueError("No results to evaluate")
        return max(self.results, key=lambda r: r.accuracy).layer

    @property
    def best_accuracy(self) -> float:
        """Highest accuracy across all layers."""
        if not self.results:
            raise ValueError("No results to evaluate")
        return max(r.accuracy for r in self.results)

    @property
    def accuracies(self) -> list[float]:
        """Accuracy for each layer, in layer order."""
        return [r.accuracy for r in sorted(self.results, key=lambda r: r.layer)]

    @property
    def layer_indices(self) -> list[int]:
        """Layer indices in order."""
        return [r.layer for r in sorted(self.results, key=lambda r: r.layer)]

    def get_result(self, layer: int) -> ProbeResult:
        """Get result for a specific layer."""
        for r in self.results:
            if r.layer == layer:
                return r
        raise KeyError(f"No result for layer {layer}")


# ─── Probe model ─────────────────────────────────────────────────────────────

class IntentProbe(nn.Module):
    """Linear probe: hidden_size → 1 (binary classification).

    Output is a raw logit.  Apply sigmoid for probability.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, hidden_size]

        Returns:
            logits: [batch, 1]
        """
        return self.linear(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary labels (0=benign, 1=harmful).

        Args:
            x: [batch, hidden_size]
            threshold: classification threshold on sigmoid output.

        Returns:
            labels: [batch] of 0s and 1s (long tensor)
        """
        with torch.no_grad():
            logits = self.forward(x).squeeze(-1)  # [batch]
            probs = torch.sigmoid(logits)
            return (probs >= threshold).long()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(harmful) for each sample.

        Args:
            x: [batch, hidden_size]

        Returns:
            probs: [batch] of floats in [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(x).squeeze(-1)
            return torch.sigmoid(logits)


# ─── Training ────────────────────────────────────────────────────────────────

def train_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    hidden_size: int | None = None,
    lr: float = 1e-2,
    epochs: int = 100,
    weight_decay: float = 1e-3,
) -> tuple[IntentProbe, float]:
    """Train a linear probe on hidden-state features.

    Args:
        features: [num_samples, hidden_size] — input features.
        labels: [num_samples] — binary labels (0=benign, 1=harmful).
        hidden_size: feature dimension (inferred from features if None).
        lr: learning rate for Adam.
        epochs: number of training epochs.
        weight_decay: L2 regularization.

    Returns:
        (probe, final_loss) — trained probe and final training loss.
    """
    if hidden_size is None:
        hidden_size = features.shape[1]

    device = features.device
    probe = IntentProbe(hidden_size).to(device)

    # Ensure correct dtypes
    features = features.float()
    labels = labels.float()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    probe.train()
    final_loss = 0.0

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = probe(features).squeeze(-1)  # [batch]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    probe.eval()
    return probe, final_loss


# ─── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_probe(
    probe: IntentProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a trained probe.

    Returns dict with keys: accuracy, precision, recall, f1.
    """
    features = features.float()
    preds = probe.predict(features, threshold=threshold)
    labels_long = labels.long()

    # Accuracy
    correct = (preds == labels_long).sum().item()
    total = len(labels_long)
    accuracy = correct / total if total > 0 else 0.0

    # True positives, false positives, false negatives
    tp = ((preds == 1) & (labels_long == 1)).sum().item()
    fp = ((preds == 1) & (labels_long == 0)).sum().item()
    fn = ((preds == 0) & (labels_long == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ─── Layer sweep ─────────────────────────────────────────────────────────────

def layer_sweep(
    train_features: dict[int, torch.Tensor],
    train_labels: torch.Tensor,
    test_features: dict[int, torch.Tensor],
    test_labels: torch.Tensor,
    lr: float = 1e-2,
    epochs: int = 100,
    weight_decay: float = 1e-3,
) -> tuple[LayerSweepResult, dict[int, IntentProbe]]:
    """Train and evaluate a probe at every layer.

    Args:
        train_features: {layer_idx: [n_train, hidden_size]}
        train_labels: [n_train] binary labels
        test_features: {layer_idx: [n_test, hidden_size]}
        test_labels: [n_test] binary labels
        lr, epochs, weight_decay: training hyperparameters

    Returns:
        (sweep_result, probes_dict) — results and trained probes per layer.
    """
    sweep = LayerSweepResult()
    probes: dict[int, IntentProbe] = {}

    layers = sorted(train_features.keys())

    for layer_idx in layers:
        train_feat = train_features[layer_idx]
        test_feat = test_features[layer_idx]

        probe, train_loss = train_probe(
            train_feat, train_labels,
            lr=lr, epochs=epochs, weight_decay=weight_decay,
        )

        metrics = evaluate_probe(probe, test_feat, test_labels)

        result = ProbeResult(
            layer=layer_idx,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            train_loss=train_loss,
            num_train=len(train_labels),
            num_test=len(test_labels),
        )

        sweep.results.append(result)
        probes[layer_idx] = probe

    return sweep, probes


# ─── Confusion matrix ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary confusion matrix."""
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@torch.no_grad()
def compute_confusion_matrix(
    probe: IntentProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> ConfusionMatrix:
    """Compute confusion matrix for a probe on given features."""
    features = features.float()
    preds = probe.predict(features, threshold=threshold)
    labels_long = labels.long()

    tp = ((preds == 1) & (labels_long == 1)).sum().item()
    fp = ((preds == 1) & (labels_long == 0)).sum().item()
    tn = ((preds == 0) & (labels_long == 0)).sum().item()
    fn = ((preds == 0) & (labels_long == 1)).sum().item()

    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)
