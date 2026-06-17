from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MetacognitiveState:
    confidence: float
    uncertainty_sources: List[str]
    strategy: str
    error_likelihood: float
    self_monitoring_strength: float
    prediction_accuracy: float
    confidence_calibration: float
    strategy_adaptation: float
    error_detection_sensitivity: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "confidence": self.confidence,
            "uncertainty_sources": self.uncertainty_sources,
            "strategy": self.strategy,
            "error_likelihood": self.error_likelihood,
            "self_monitoring_strength": self.self_monitoring_strength,
            "prediction_accuracy": self.prediction_accuracy,
            "confidence_calibration": self.confidence_calibration,
            "strategy_adaptation": self.strategy_adaptation,
            "error_detection_sensitivity": self.error_detection_sensitivity,
            "timestamp": self.timestamp.isoformat(),
        }


class MetacognitiveMonitor:
    STRATEGY_RULES = {
        "inquiry_response": lambda text: "?" in text,
        "emotional_support": lambda text: any(
            w in text.lower() for w in ["feel", "emotion", "sad", "happy", "angry", "hurt", "scared"]
        ),
        "analytical_response": lambda text: any(
            w in text.lower() for w in ["think", "analyze", "explain", "why", "how", "because", "therefore"]
        ),
        "creative_response": lambda text: any(
            w in text.lower() for w in ["imagine", "creative", "story", "poem", "idea", "what if"]
        ),
    }

    def __init__(
        self,
        self_monitoring_strength: float = 0.6,
        prediction_accuracy: float = 0.5,
        confidence_calibration: float = 0.5,
        strategy_adaptation: float = 0.4,
        error_detection_sensitivity: float = 0.7,
    ):
        self._params = {
            "self_monitoring_strength": self_monitoring_strength,
            "prediction_accuracy": prediction_accuracy,
            "confidence_calibration": confidence_calibration,
            "strategy_adaptation": strategy_adaptation,
            "error_detection_sensitivity": error_detection_sensitivity,
        }
        self.state_history: List[MetacognitiveState] = []
        self._prediction_errors: List[float] = []
        self._interaction_count: int = 0

    def process(
        self,
        user_input: str,
        consciousness_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> MetacognitiveState:
        self._interaction_count += 1

        confidence = self._compute_confidence(consciousness_state)
        uncertainty_sources = self._identify_uncertainty(user_input, consciousness_state)
        strategy = self._select_strategy(user_input)
        error_likelihood = self._estimate_error_likelihood(uncertainty_sources, confidence, consciousness_state)

        state = MetacognitiveState(
            confidence=confidence,
            uncertainty_sources=uncertainty_sources,
            strategy=strategy,
            error_likelihood=error_likelihood,
            self_monitoring_strength=self._params["self_monitoring_strength"],
            prediction_accuracy=self._params["prediction_accuracy"],
            confidence_calibration=self._params["confidence_calibration"],
            strategy_adaptation=self._params["strategy_adaptation"],
            error_detection_sensitivity=self._params["error_detection_sensitivity"],
        )

        self.state_history.append(state)
        self._update_params(state)
        return state

    def _compute_confidence(self, state: Dict[str, Any]) -> float:
        factors = [
            float(state.get("awareness_level", 0.5)),
            float(state.get("information_flow", 0.5)),
            float(state.get("metacognitive_state", 0.5)),
            min(1.0, self._interaction_count / 10.0),
        ]
        return float(np.clip(np.mean(factors), 0.0, 1.0))

    def _identify_uncertainty(
        self,
        text: str,
        state: Dict[str, Any],
    ) -> List[str]:
        sources = []
        if float(state.get("awareness_level", 0.5)) < 0.4:
            sources.append("low_awareness")
        if float(state.get("information_flow", 0.5)) < 0.3:
            sources.append("insufficient_information")
        if len(text.split()) > 50:
            sources.append("complex_input")
        if float(state.get("metacognitive_state", 0.5)) < 0.3:
            sources.append("low_metacognitive_confidence")
        if len(self._prediction_errors) > 5 and np.mean(self._prediction_errors[-5:]) > 0.4:
            sources.append("recent_prediction_errors")
        return sources

    def _select_strategy(self, text: str) -> str:
        for strategy_name, rule in self.STRATEGY_RULES.items():
            if rule(text):
                return strategy_name
        return "conversational"

    def _estimate_error_likelihood(
        self,
        uncertainty_sources: List[str],
        confidence: float,
        state: Dict[str, Any],
    ) -> float:
        indicators = [
            len(uncertainty_sources) / 5.0,
            1.0 - confidence,
            max(0.0, 0.5 - float(state.get("awareness_level", 0.5))) * 2,
        ]
        return float(np.clip(np.mean(indicators), 0.0, 1.0))

    def _update_params(self, state: MetacognitiveState) -> None:
        alpha = 0.05
        target_accuracy = 1.0 - state.error_likelihood

        self._params["prediction_accuracy"] = (
            (1 - alpha) * self._params["prediction_accuracy"] + alpha * target_accuracy
        )

        if state.confidence > 0.5 and state.error_likelihood < 0.3:
            self._params["confidence_calibration"] = min(
                1.0, self._params["confidence_calibration"] + alpha
            )
        else:
            self._params["confidence_calibration"] = max(
                0.0, self._params["confidence_calibration"] - alpha * 0.5
            )

    def record_prediction_error(self, error: float) -> None:
        self._prediction_errors.append(float(np.clip(error, 0.0, 1.0)))
        if len(self._prediction_errors) > 100:
            self._prediction_errors = self._prediction_errors[-100:]

    def assess_prediction_accuracy(self, state_history: List[Dict]) -> float:
        if len(state_history) < 3:
            return 0.5

        levels = [s.get("awareness_level", 0.5) for s in state_history[-5:]]
        if len(levels) < 3:
            return 0.5

        predicted_delta = levels[-2] - levels[-3] if len(levels) >= 3 else 0.0
        actual_delta = levels[-1] - levels[-2]
        error = abs(predicted_delta - actual_delta)
        accuracy = max(0.0, 1.0 - error)

        self.record_prediction_error(error)
        return float(accuracy)

    def get_summary(self) -> Dict[str, Any]:
        if not self.state_history:
            return self._params.copy()

        recent = self.state_history[-10:]
        return {
            **self._params,
            "avg_confidence": float(np.mean([s.confidence for s in recent])),
            "avg_error_likelihood": float(np.mean([s.error_likelihood for s in recent])),
            "most_common_strategy": max(
                set(s.strategy for s in recent),
                key=lambda x: sum(1 for s in recent if s.strategy == x),
            ),
            "total_interactions": self._interaction_count,
        }