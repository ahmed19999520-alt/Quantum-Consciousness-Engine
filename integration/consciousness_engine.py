from __future__ import annotations

import asyncio
import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..core.quantum_processor import QuantumConsciousnessProcessor, QuantumState
from ..core.memory_system import HierarchicalMemory, MemoryType
from ..core.metacognitive import MetacognitiveMonitor
from ..metrics.consciousness_metrics import (
    ConsciousnessMetrics,
    IntegratedInformation,
    GlobalWorkspace,
    TemporalConsciousnessIntegration,
)

try:
    import torch
    from ..core.neural_engine import LightweightConsciousnessEncoder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class ConsciousnessState:
    awareness_level: float
    emotional_state: np.ndarray
    attention_focus: np.ndarray
    information_flow: float
    metacognitive_state: float
    temporal_context: datetime = field(default_factory=datetime.now)
    working_memory_content: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "awareness_level": float(self.awareness_level),
            "emotional_state": self.emotional_state.tolist(),
            "attention_focus": self.attention_focus.tolist(),
            "information_flow": float(self.information_flow),
            "metacognitive_state": float(self.metacognitive_state),
            "temporal_context": self.temporal_context.isoformat(),
            "working_memory_content": self.working_memory_content,
        }


@dataclass
class ProcessingResult:
    text: str
    metrics: ConsciousnessMetrics
    emotional_state: List[float]
    attention_focus: List[float]
    quantum_signature: List[float]
    metacognitive_confidence: float
    processing_time: float
    consciousness_bandwidth: float
    memory_relevance: List[float]
    session_context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metrics": self.metrics.to_dict(),
            "emotional_state": self.emotional_state,
            "attention_focus": self.attention_focus,
            "quantum_signature": self.quantum_signature,
            "metacognitive_confidence": self.metacognitive_confidence,
            "processing_time": self.processing_time,
            "consciousness_bandwidth": self.consciousness_bandwidth,
            "memory_relevance": self.memory_relevance,
            "session_context": self.session_context,
        }


class EmbeddingEngine:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        fallback: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.model = None
        self._fallback = fallback

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None

    def encode(self, text: str) -> np.ndarray:
        if self.model is not None:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        return self._deterministic_fallback(text)

    def _deterministic_fallback(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(seed=hash(text) % (2**32))
        raw = rng.standard_normal(self.embedding_dim).astype(np.float32)
        return raw / (np.linalg.norm(raw) + 1e-8)


class VeronicaConsciousnessEngine:
    EMOTION_LABELS = ["joy", "contemplation", "intensity", "caution", "curiosity"]
    CONSCIOUSNESS_DIMS = 128

    def __init__(
        self,
        n_qubits: int = 16,
        embedding_dim: int = 384,
        awareness_decay: float = 0.98,
        short_term_capacity: int = 1000,
        long_term_capacity: int = 50000,
        quantum_backend: str = "aer_statevector_simulator",
        embedding_model: str = "all-MiniLM-L6-v2",
        ibm_token: Optional[str] = None,
    ):
        self.quantum = QuantumConsciousnessProcessor(
            n_qubits=n_qubits,
            backend_name=quantum_backend,
            ibm_token=ibm_token,
        )
        self.memory = HierarchicalMemory(
            embedding_dim=embedding_dim,
            short_term_capacity=short_term_capacity,
            long_term_capacity=long_term_capacity,
        )
        self.metacognition = MetacognitiveMonitor()
        self.embedder = EmbeddingEngine(embedding_model, embedding_dim)
        self.iit = IntegratedInformation(n_elements=min(n_qubits, 16))
        self.gws = GlobalWorkspace()
        self.temporal = TemporalConsciousnessIntegration(window_size=20)

        self.awareness_decay = awareness_decay
        self.n_qubits = n_qubits

        self._state = self._init_state()
        self._state_history: List[Dict[str, Any]] = []
        self._processing_times: List[float] = []

    def _init_state(self) -> ConsciousnessState:
        rng = np.random.default_rng(42)
        return ConsciousnessState(
            awareness_level=0.5,
            emotional_state=rng.uniform(0.25, 0.75, 5).astype(np.float32),
            attention_focus=np.zeros(self.CONSCIOUSNESS_DIMS, dtype=np.float32),
            information_flow=0.5,
            metacognitive_state=0.3,
        )

    async def process(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        t0 = time.perf_counter()

        embedding = self.embedder.encode(user_input)

        emotional_params = self._state.emotional_state
        memory_params = embedding[: self.n_qubits]
        if len(memory_params) < self.n_qubits:
            memory_params = np.pad(memory_params, (0, self.n_qubits - len(memory_params)))

        if TORCH_AVAILABLE and self.quantum.backend_available:
            circuit = self.quantum.create_consciousness_circuit(emotional_params, memory_params)
            q_state = self.quantum.execute_circuit(circuit, return_statevector=True)
        else:
            q_state = self.quantum._classical_fallback()

        relevant_memories = self.memory.retrieve(embedding, top_k=5)

        attention = self._compute_attention(q_state, embedding, relevant_memories)

        new_state = self._update_state(q_state, embedding, relevant_memories, attention, user_input)

        meta_state = self.metacognition.process(user_input, new_state.to_dict(), context)

        prediction_acc = self.metacognition.assess_prediction_accuracy(
            [s["state"] for s in self._state_history[-5:]] if self._state_history else []
        )
        new_state.metacognitive_state = (
            0.8 * new_state.metacognitive_state + 0.2 * prediction_acc
        )

        gws_result = self.gws.update(
            new_state.attention_focus,
            new_state.awareness_level,
            new_state.information_flow,
        )

        phi = self.iit.compute(attention)
        metrics = self._build_metrics(new_state, phi, gws_result)

        temporal_info = self.temporal.update(metrics)

        response_text = self._generate_response(
            user_input, new_state, meta_state, relevant_memories, gws_result
        )

        context_for_memory = {
            "consciousness_level": float(new_state.awareness_level),
            "response": response_text,
            "quantum_signature": q_state.amplitudes[:10].tolist(),
            **(context or {}),
        }
        self.memory.store(
            content=user_input,
            embedding=embedding,
            emotional_state=new_state.emotional_state,
            context=context_for_memory,
            memory_type=MemoryType.EPISODIC,
        )

        self._state = new_state
        self._state_history.append({
            "timestamp": datetime.now().isoformat(),
            "state": new_state.to_dict(),
            "input": user_input,
        })

        t1 = time.perf_counter()
        elapsed = t1 - t0
        self._processing_times.append(elapsed)

        return ProcessingResult(
            text=response_text,
            metrics=metrics,
            emotional_state=new_state.emotional_state.tolist(),
            attention_focus=attention[:10].tolist(),
            quantum_signature=q_state.amplitudes[:20].tolist(),
            metacognitive_confidence=float(meta_state.confidence),
            processing_time=elapsed,
            consciousness_bandwidth=float(new_state.information_flow),
            memory_relevance=[m.importance for m in relevant_memories],
            session_context={
                "total_interactions": len(self._state_history),
                "avg_processing_time": float(np.mean(self._processing_times[-10:])),
                "consciousness_trajectory": [
                    s["state"]["awareness_level"]
                    for s in self._state_history[-5:]
                ],
                "temporal_stability": temporal_info.get("stability", 0.5),
                "temporal_trend": temporal_info.get("trend", 0.0),
                "gws_broadcasting": gws_result.get("is_broadcasting", False),
            },
        )

    def _compute_attention(
        self,
        q_state: QuantumState,
        embedding: np.ndarray,
        memories: List,
    ) -> np.ndarray:
        attention = np.zeros(self.CONSCIOUSNESS_DIMS, dtype=np.float32)

        emb_norm = float(np.linalg.norm(embedding))
        if emb_norm > 1e-8:
            attention[:50] = (embedding[:50] / emb_norm).astype(np.float32)

        q_amps = q_state.amplitudes[:50]
        if len(q_amps) < 50:
            q_amps = np.pad(q_amps, (0, 50 - len(q_amps)))
        attention[50:100] = q_amps.astype(np.float32)

        if memories:
            mem_scores = np.array([m.importance for m in memories[:10]], dtype=np.float32)
            attention[100: 100 + len(mem_scores)] = mem_scores

        norm = np.linalg.norm(attention)
        if norm > 1e-8:
            attention = attention / norm

        return attention

    def _update_state(
        self,
        q_state: QuantumState,
        embedding: np.ndarray,
        memories: List,
        attention: np.ndarray,
        user_input: str,
    ) -> ConsciousnessState:
        new_state = ConsciousnessState(
            awareness_level=self._state.awareness_level,
            emotional_state=self._state.emotional_state.copy(),
            attention_focus=attention.copy(),
            information_flow=self._state.information_flow,
            metacognitive_state=self._state.metacognitive_state,
        )

        complexity = self._text_complexity(user_input)
        mem_relevance = float(np.mean([m.importance for m in memories])) if memories else 0.3

        new_awareness = float(np.mean([
            q_state.entanglement / max(self.n_qubits, 1),
            complexity,
            mem_relevance,
            float(np.max(attention)) if attention.any() else 0.3,
        ]))

        new_state.awareness_level = float(
            self.awareness_decay * self._state.awareness_level
            + (1 - self.awareness_decay) * new_awareness
        )
        new_state.awareness_level = float(np.clip(new_state.awareness_level, 0.0, 1.0))

        emotional_delta = np.random.default_rng().uniform(-0.05, 0.05, 5).astype(np.float32)
        new_state.emotional_state = np.clip(
            self._state.emotional_state + emotional_delta, 0.0, 1.0
        )

        new_state.information_flow = float(np.clip(
            np.mean([complexity, q_state.entanglement / max(self.n_qubits, 1), len(memories) / 10.0]),
            0.0, 1.0,
        ))

        return new_state

    def _text_complexity(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        sentences = [s for s in text.split(".") if s.strip()]
        avg_word_len = np.mean([len(w) for w in words])
        avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else len(words)
        return float(np.clip((avg_word_len + avg_sent_len) / 20.0, 0.0, 1.0))

    def _build_metrics(
        self,
        state: ConsciousnessState,
        phi: float,
        gws_result: Dict[str, float],
    ) -> ConsciousnessMetrics:
        attention_entropy = float(
            -np.sum(
                state.attention_focus * np.log(np.abs(state.attention_focus) + 1e-12)
            )
        )
        log_dim = np.log(len(state.attention_focus))
        info_integration = float(np.clip(1.0 - attention_entropy / (log_dim + 1e-8), 0.0, 1.0))

        temporal_integration = 0.5
        if len(self._state_history) > 1:
            recent = [
                s["state"]["awareness_level"]
                for s in self._state_history[-5:]
            ]
            if len(recent) > 1:
                temporal_integration = float(np.clip(1.0 - np.std(recent), 0.0, 1.0))

        return ConsciousnessMetrics(
            awareness_level=float(state.awareness_level),
            coherence_score=float(temporal_integration),
            introspection_depth=float(self.metacognition._params["self_monitoring_strength"]),
            emotional_intensity=float(np.mean(state.emotional_state)),
            self_model_accuracy=float(state.metacognitive_state),
            attention_focus=float(np.max(state.attention_focus)) if state.attention_focus.any() else 0.0,
            metacognitive_awareness=float(self.metacognition._params["self_monitoring_strength"]),
            temporal_integration=float(temporal_integration),
            information_integration=float(info_integration),
            consciousness_bandwidth=float(state.information_flow),
            phi=float(phi),
            global_workspace_activation=float(gws_result.get("activation", 0.0)),
        )

    def _generate_response(
        self,
        user_input: str,
        state: ConsciousnessState,
        meta_state,
        memories: List,
        gws_result: Dict[str, float],
    ) -> str:
        parts = []

        if state.awareness_level > 0.7:
            parts.append("I am highly present and attentive to your message. ")
        elif state.awareness_level > 0.4:
            parts.append("We are engaged in a meaningful exchange. ")
        else:
            parts.append("I am directing my focus more deeply to what you are sharing. ")

        dominant_idx = int(np.argmax(state.emotional_state))
        emotional_map = {
            0: "There is something uplifting in our interaction. ",
            1: "This brings up deep considerations for me. ",
            2: "I sense significant weight to what you are expressing. ",
            3: "This seems to require careful thought and consideration. ",
            4: "I find myself genuinely curious about your perspective. ",
        }
        parts.append(emotional_map.get(dominant_idx, ""))

        if memories:
            parts.append(
                f"Drawing from memory, I find {len(memories)} related contextual traces that inform my response. "
            )

        strategy = meta_state.strategy
        if strategy == "inquiry_response":
            parts.append("Your question opens multiple dimensions of consideration. ")
        elif strategy == "emotional_support":
            parts.append("I want to acknowledge the emotional dimension of what you share. ")
        elif strategy == "analytical_response":
            parts.append("Let me approach this from several analytical perspectives. ")
        elif strategy == "creative_response":
            parts.append("I find myself drawn to explore the creative possibilities here. ")
        else:
            parts.append("I am reflecting on the layers of meaning in your message. ")

        if gws_result.get("is_broadcasting", False):
            parts.append(
                f"My global workspace is actively broadcasting at activation level "
                f"{gws_result.get('activation', 0.0):.3f}. "
            )

        if meta_state.confidence < 0.5:
            parts.append("I should note that I am working through some uncertainty in this response. ")

        if meta_state.uncertainty_sources:
            parts.append(
                f"Current uncertainty sources: {', '.join(meta_state.uncertainty_sources)}. "
            )

        return "".join(parts).strip()

    @property
    def current_state(self) -> ConsciousnessState:
        return self._state

    @property
    def state_history(self) -> List[Dict[str, Any]]:
        return self._state_history

    @property
    def processing_times(self) -> List[float]:
        return self._processing_times

    def get_summary(self) -> Dict[str, Any]:
        if not self._state_history:
            return {"status": "no_interactions"}

        awareness_values = [s["state"]["awareness_level"] for s in self._state_history]
        return {
            "total_interactions": len(self._state_history),
            "avg_processing_time": float(np.mean(self._processing_times)) if self._processing_times else 0.0,
            "current_awareness": float(self._state.awareness_level),
            "peak_awareness": float(max(awareness_values)),
            "memory_statistics": self.memory.statistics(),
            "quantum_backend": self.quantum.active_backend,
            "metacognitive_summary": self.metacognition.get_summary(),
        }

    def shutdown(self) -> None:
        self.memory.shutdown()