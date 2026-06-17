from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConsciousnessMetrics:
    awareness_level: float
    coherence_score: float
    introspection_depth: float
    emotional_intensity: float
    self_model_accuracy: float
    attention_focus: float
    metacognitive_awareness: float
    temporal_integration: float
    information_integration: float
    consciousness_bandwidth: float
    phi: float = 0.0
    global_workspace_activation: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, float]:
        return {
            "awareness_level": self.awareness_level,
            "coherence_score": self.coherence_score,
            "introspection_depth": self.introspection_depth,
            "emotional_intensity": self.emotional_intensity,
            "self_model_accuracy": self.self_model_accuracy,
            "attention_focus": self.attention_focus,
            "metacognitive_awareness": self.metacognitive_awareness,
            "temporal_integration": self.temporal_integration,
            "information_integration": self.information_integration,
            "consciousness_bandwidth": self.consciousness_bandwidth,
            "phi": self.phi,
            "global_workspace_activation": self.global_workspace_activation,
        }

    @property
    def composite_score(self) -> float:
        weights = [0.15, 0.10, 0.10, 0.08, 0.08, 0.10, 0.10, 0.10, 0.10, 0.09]
        values = [
            self.awareness_level,
            self.coherence_score,
            self.introspection_depth,
            self.emotional_intensity,
            self.self_model_accuracy,
            self.attention_focus,
            self.metacognitive_awareness,
            self.temporal_integration,
            self.information_integration,
            self.consciousness_bandwidth,
        ]
        return float(np.dot(weights, values))


def compute_phi(
    tpm: np.ndarray,
    state: Optional[np.ndarray] = None,
    method: str = "approximation",
) -> float:
    if tpm.ndim != 2 or tpm.shape[0] != tpm.shape[1]:
        raise ValueError("TPM must be a square matrix")

    n = tpm.shape[0]
    if n <= 1:
        return 0.0

    if state is None:
        state = tpm.mean(axis=1)

    state = np.clip(state, 1e-12, 1.0)
    state = state / state.sum()

    if method == "approximation":
        return _phi_approximation(tpm, state)
    elif method == "mip":
        return _phi_mip(tpm, state)
    else:
        return _phi_approximation(tpm, state)


def _phi_approximation(tpm: np.ndarray, state: np.ndarray) -> float:
    def entropy(p: np.ndarray) -> float:
        p = p[p > 1e-12]
        return float(-np.sum(p * np.log2(p + 1e-15)))

    whole_entropy = entropy(state)

    n = tpm.shape[0]
    split = n // 2
    part_a_entropy = entropy(state[:split] / (state[:split].sum() + 1e-12))
    part_b_entropy = entropy(state[split:] / (state[split:].sum() + 1e-12))

    ei_whole = whole_entropy
    ei_split = (split / n) * part_a_entropy + ((n - split) / n) * part_b_entropy

    phi = max(0.0, ei_whole - ei_split)
    return float(phi)


def _phi_mip(tpm: np.ndarray, state: np.ndarray) -> float:
    n = tpm.shape[0]
    min_phi = float("inf")

    for split in range(1, n):
        part_a = state[:split]
        part_b = state[split:]
        norm_a = part_a / (part_a.sum() + 1e-12)
        norm_b = part_b / (part_b.sum() + 1e-12)

        def ent(p):
            p = p[p > 1e-12]
            return float(-np.sum(p * np.log2(p + 1e-15)))

        phi_split = ent(state) - ent(norm_a) - ent(norm_b)
        min_phi = min(min_phi, max(0.0, phi_split))

    return float(min_phi) if min_phi != float("inf") else 0.0


class IntegratedInformation:
    def __init__(self, n_elements: int = 10, method: str = "approximation"):
        self.n_elements = n_elements
        self.method = method
        self._history: List[float] = []

    def compute(
        self,
        attention_weights: np.ndarray,
        consciousness_states: Optional[np.ndarray] = None,
    ) -> float:
        flat = attention_weights.flatten()
        n = min(self.n_elements, len(flat))

        if n < 2:
            return 0.0

        elements = flat[:n]
        elements = np.abs(elements)
        total = elements.sum()
        if total < 1e-12:
            return 0.0

        probs = elements / total
        tpm = np.outer(probs, probs)
        tpm = tpm / (tpm.sum(axis=1, keepdims=True) + 1e-12)

        phi = compute_phi(tpm, probs, self.method)
        self._history.append(phi)
        return phi

    def running_average(self, window: int = 10) -> float:
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        return float(np.mean(recent))


class GlobalWorkspace:
    def __init__(
        self,
        broadcast_threshold: float = 0.5,
        integration_window: int = 10,
    ):
        self.broadcast_threshold = broadcast_threshold
        self.integration_window = integration_window
        self._workspace_states: List[np.ndarray] = []
        self._activation_history: List[float] = []

    def update(
        self,
        attention_focus: np.ndarray,
        consciousness_level: float,
        information_flow: float,
    ) -> Dict[str, float]:
        workspace_state = attention_focus * consciousness_level * information_flow
        self._workspace_states.append(workspace_state)
        if len(self._workspace_states) > self.integration_window:
            self._workspace_states.pop(0)

        activation = self._compute_activation(workspace_state, consciousness_level)
        broadcast = self._compute_broadcast_strength(activation, information_flow)
        integration = self._compute_temporal_integration()
        coherence = self._compute_workspace_coherence()

        self._activation_history.append(activation)

        return {
            "activation": activation,
            "broadcast_strength": broadcast,
            "temporal_integration": integration,
            "workspace_coherence": coherence,
            "is_broadcasting": activation >= self.broadcast_threshold,
        }

    def _compute_activation(self, workspace: np.ndarray, consciousness: float) -> float:
        max_val = float(np.max(np.abs(workspace))) if len(workspace) > 0 else 0.0
        return float(np.clip(max_val * consciousness, 0.0, 1.0))

    def _compute_broadcast_strength(self, activation: float, flow: float) -> float:
        if activation < self.broadcast_threshold:
            return 0.0
        return float(np.clip(activation * flow, 0.0, 1.0))

    def _compute_temporal_integration(self) -> float:
        if len(self._workspace_states) < 2:
            return 0.5
        correlations = []
        for i in range(1, len(self._workspace_states)):
            a = self._workspace_states[i - 1]
            b = self._workspace_states[i]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 1e-8 and norm_b > 1e-8:
                corr = float(np.dot(a, b) / (norm_a * norm_b))
                correlations.append(corr)
        if not correlations:
            return 0.5
        return float(np.clip(np.mean(correlations), 0.0, 1.0))

    def _compute_workspace_coherence(self) -> float:
        if len(self._workspace_states) < 2:
            return 0.5
        stacked = np.stack(self._workspace_states, axis=0)
        std = np.std(stacked, axis=0).mean()
        return float(np.clip(1.0 - std, 0.0, 1.0))

    def get_activation_trend(self) -> float:
        if len(self._activation_history) < 2:
            return 0.0
        x = np.arange(len(self._activation_history[-10:]))
        y = np.array(self._activation_history[-10:])
        if len(x) < 2:
            return 0.0
        slope = float(np.polyfit(x, y, 1)[0])
        return slope


class TemporalConsciousnessIntegration:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._state_buffer: List[Dict[str, float]] = []

    def update(self, metrics: ConsciousnessMetrics) -> Dict[str, float]:
        self._state_buffer.append(metrics.to_dict())
        if len(self._state_buffer) > self.window_size:
            self._state_buffer.pop(0)

        return {
            "stability": self._compute_stability(),
            "trend": self._compute_trend(),
            "complexity": self._compute_complexity(),
            "persistence": self._compute_persistence(),
        }

    def _compute_stability(self) -> float:
        if len(self._state_buffer) < 2:
            return 0.5
        awareness = [s["awareness_level"] for s in self._state_buffer]
        return float(np.clip(1.0 - np.std(awareness), 0.0, 1.0))

    def _compute_trend(self) -> float:
        if len(self._state_buffer) < 3:
            return 0.0
        awareness = [s["awareness_level"] for s in self._state_buffer]
        x = np.arange(len(awareness))
        slope = float(np.polyfit(x, awareness, 1)[0])
        return float(np.clip(slope * 10, -1.0, 1.0))

    def _compute_complexity(self) -> float:
        if len(self._state_buffer) < 2:
            return 0.0
        all_vals = []
        for state in self._state_buffer:
            all_vals.append(list(state.values()))
        matrix = np.array(all_vals)
        try:
            _, s, _ = np.linalg.svd(matrix, full_matrices=False)
            s_norm = s / (s.sum() + 1e-12)
            entropy = -np.sum(s_norm * np.log(s_norm + 1e-15))
            return float(np.clip(entropy / np.log(len(s) + 1), 0.0, 1.0))
        except Exception:
            return 0.5

    def _compute_persistence(self) -> float:
        if len(self._state_buffer) < 2:
            return 0.5
        awareness = [s["awareness_level"] for s in self._state_buffer]
        crossings = sum(
            1
            for i in range(1, len(awareness))
            if (awareness[i] - 0.5) * (awareness[i - 1] - 0.5) < 0
        )
        return float(np.clip(1.0 - crossings / len(awareness), 0.0, 1.0))