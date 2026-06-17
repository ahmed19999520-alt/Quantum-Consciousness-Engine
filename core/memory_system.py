from __future__ import annotations

import hashlib
import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    EMOTIONAL = "emotional"


@dataclass
class MemoryTrace:
    content: str
    embedding: np.ndarray
    emotional_tags: List[str]
    importance: float
    memory_type: MemoryType
    context_hash: str
    access_count: int = 0
    consolidation_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

    def update_access(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.now()

    def temporal_relevance(self, decay_days: float = 30.0) -> float:
        elapsed = (datetime.now() - self.last_accessed).total_seconds() / 86400.0
        return float(np.exp(-elapsed / decay_days))

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "emotional_tags": self.emotional_tags,
            "importance": self.importance,
            "memory_type": self.memory_type.value,
            "context_hash": self.context_hash,
            "access_count": self.access_count,
            "consolidation_score": self.consolidation_score,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
        }


class VectorIndex:
    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self._use_faiss = use_faiss and FAISS_AVAILABLE
        self._index = None
        self._embeddings: List[np.ndarray] = []
        self._init_index()

    def _init_index(self) -> None:
        if self._use_faiss:
            self._index = faiss.IndexFlatIP(self.dim)
        else:
            self._embeddings = []

    def add(self, embedding: np.ndarray) -> None:
        vec = self._normalize(embedding)
        if self._use_faiss:
            self._index.add(vec.reshape(1, -1).astype(np.float32))
        else:
            self._embeddings.append(vec)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        vec = self._normalize(query)
        k = min(k, self.size)
        if k == 0:
            return np.array([[]]), np.array([[]])

        if self._use_faiss:
            scores, indices = self._index.search(
                vec.reshape(1, -1).astype(np.float32), k
            )
            return scores[0], indices[0]
        else:
            scores = np.array([
                float(np.dot(vec, emb)) for emb in self._embeddings
            ])
            indices = np.argsort(scores)[::-1][:k]
            return scores[indices], indices

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def reset(self) -> None:
        self._init_index()
        self._embeddings = []

    @property
    def size(self) -> int:
        if self._use_faiss:
            return self._index.ntotal
        return len(self._embeddings)


class HierarchicalMemory:
    EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise"]
    EMOTION_THRESHOLD = 0.6
    WORKING_MEMORY_LIMIT = 20
    CONSOLIDATION_RATIO = 0.5

    def __init__(
        self,
        embedding_dim: int = 384,
        short_term_capacity: int = 1000,
        long_term_capacity: int = 50000,
        consolidation_threshold: float = 0.75,
        decay_rate: float = 0.99,
        consolidation_interval: int = 300,
    ):
        self.embedding_dim = embedding_dim
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        self.consolidation_threshold = consolidation_threshold
        self.decay_rate = decay_rate
        self.consolidation_interval = consolidation_interval

        self.short_term: List[MemoryTrace] = []
        self.long_term: List[MemoryTrace] = []
        self.working: List[MemoryTrace] = []
        self.index = VectorIndex(embedding_dim)
        self.index_to_memory: List[int] = []

        self._lock = threading.RLock()
        self._consolidation_thread: Optional[threading.Thread] = None
        self._running = False
        self._start_consolidation_loop()

    def _start_consolidation_loop(self) -> None:
        self._running = True

        def loop():
            while self._running:
                time.sleep(self.consolidation_interval)
                try:
                    self.consolidate()
                except Exception:
                    pass

        self._consolidation_thread = threading.Thread(target=loop, daemon=True)
        self._consolidation_thread.start()

    def store(
        self,
        content: str,
        embedding: np.ndarray,
        emotional_state: np.ndarray,
        context: Dict[str, Any],
        memory_type: MemoryType = MemoryType.EPISODIC,
    ) -> str:
        with self._lock:
            emb = self._prepare_embedding(embedding)
            tags = self._extract_emotional_tags(emotional_state)
            importance = self._calculate_importance(content, emotional_state, context)
            context_hash = self._hash_context(context)

            trace = MemoryTrace(
                content=content,
                embedding=emb,
                emotional_tags=tags,
                importance=importance,
                memory_type=memory_type,
                context_hash=context_hash,
            )

            if memory_type == MemoryType.WORKING:
                self.working.append(trace)
                if len(self.working) > self.WORKING_MEMORY_LIMIT:
                    self.working.pop(0)
            else:
                self.short_term.append(trace)
                self.index.add(emb)
                self.index_to_memory.append(len(self.short_term) - 1)

            if len(self.short_term) >= self.short_term_capacity:
                self.consolidate()

            return context_hash

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
        time_window_days: Optional[float] = None,
    ) -> List[MemoryTrace]:
        with self._lock:
            query = self._prepare_embedding(query_embedding)

            if self.index.size == 0:
                return []

            scores, raw_indices = self.index.search(query, min(top_k * 3, self.index.size))

            results: List[Tuple[float, MemoryTrace]] = []

            for score, raw_idx in zip(scores, raw_indices):
                if raw_idx < 0 or raw_idx >= len(self.index_to_memory):
                    continue

                mem_idx = self.index_to_memory[raw_idx]
                all_memories = self.short_term + self.long_term
                if mem_idx >= len(all_memories):
                    continue

                mem = all_memories[mem_idx]

                if memory_types and mem.memory_type not in memory_types:
                    continue

                if mem.importance < min_importance:
                    continue

                if time_window_days is not None:
                    age_days = (datetime.now() - mem.created_at).total_seconds() / 86400.0
                    if age_days > time_window_days:
                        continue

                relevance = (
                    float(score) * 0.4
                    + mem.importance * 0.3
                    + min(mem.access_count / 100.0, 1.0) * 0.2
                    + mem.temporal_relevance() * 0.1
                )

                mem.update_access()
                mem.consolidation_score = relevance
                results.append((relevance, mem))

            results.sort(key=lambda x: x[0], reverse=True)
            return [mem for _, mem in results[:top_k]]

    def consolidate(self) -> Dict[str, int]:
        with self._lock:
            if not self.short_term:
                return {"consolidated": 0, "removed": 0}

            candidates = sorted(
                self.short_term, key=lambda m: m.importance, reverse=True
            )

            n_consolidate = max(1, int(len(candidates) * self.CONSOLIDATION_RATIO))
            to_consolidate = candidates[:n_consolidate]

            for mem in to_consolidate:
                mem.consolidation_score = mem.importance
                self.long_term.append(mem)

            self.short_term = [m for m in self.short_term if m not in to_consolidate]

            removed = 0
            if len(self.long_term) > self.long_term_capacity:
                overflow = len(self.long_term) - self.long_term_capacity
                self.long_term.sort(key=lambda m: m.consolidation_score)
                self.long_term = self.long_term[overflow:]
                removed = overflow

            return {"consolidated": len(to_consolidate), "removed": removed}

    def _prepare_embedding(self, embedding: np.ndarray) -> np.ndarray:
        flat = embedding.flatten()
        if len(flat) < self.embedding_dim:
            flat = np.pad(flat, (0, self.embedding_dim - len(flat)))
        return flat[: self.embedding_dim].astype(np.float32)

    def _extract_emotional_tags(self, emotional_state: np.ndarray) -> List[str]:
        tags = []
        for i, label in enumerate(self.EMOTION_LABELS):
            if i < len(emotional_state) and float(emotional_state[i]) > self.EMOTION_THRESHOLD:
                tags.append(label)
        return tags if tags else ["neutral"]

    def _calculate_importance(
        self,
        content: str,
        emotional_state: np.ndarray,
        context: Dict[str, Any],
    ) -> float:
        length_factor = min(len(content) / 1000.0, 1.0)
        question_factor = content.count("?") * 0.1
        emotional_factor = float(np.mean(emotional_state)) if len(emotional_state) > 0 else 0.3
        context_factor = float(context.get("consciousness_level", 0.5))
        novelty = 1.0 - self._similarity_to_recent(content)

        importance = (
            length_factor * 0.2
            + min(question_factor, 0.3) * 0.2
            + emotional_factor * 0.2
            + context_factor * 0.2
            + novelty * 0.2
        )
        return min(float(importance), 1.0)

    def _similarity_to_recent(self, content: str) -> float:
        words = set(content.lower().split())
        if not words or not self.short_term:
            return 0.0

        max_sim = 0.0
        for mem in self.short_term[-10:]:
            mem_words = set(mem.content.lower().split())
            if mem_words:
                overlap = len(words & mem_words)
                union = len(words | mem_words)
                if union > 0:
                    max_sim = max(max_sim, overlap / union)
        return max_sim

    def _hash_context(self, context: Dict[str, Any]) -> str:
        raw = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def statistics(self) -> Dict[str, Any]:
        with self._lock:
            all_mem = self.short_term + self.long_term
            return {
                "short_term_count": len(self.short_term),
                "long_term_count": len(self.long_term),
                "working_memory_count": len(self.working),
                "total_memories": len(all_mem),
                "avg_importance": float(np.mean([m.importance for m in all_mem])) if all_mem else 0.0,
                "index_size": self.index.size,
                "memory_types": {
                    mt.value: sum(1 for m in all_mem if m.memory_type == mt)
                    for mt in MemoryType
                },
            }

    def shutdown(self) -> None:
        self._running = False