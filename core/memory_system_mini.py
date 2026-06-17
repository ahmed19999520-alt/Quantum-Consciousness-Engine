import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

@dataclass
class MemoryTrace:
    content: str
    embedding: np.ndarray
    emotional_tags: List[str]
    importance: float
    access_count: int
    created_at: datetime
    last_accessed: datetime
    memory_type: str
    consolidation_score: float = 0.0

class HierarchicalMemory:
    def __init__(
        self,
        short_term_capacity: int = 1000,
        long_term_capacity: int = 50000,
        consolidation_threshold: float = 0.75
    ):
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        self.consolidation_threshold = consolidation_threshold
        
        self.working_memory = deque(maxlen=20)
        self.short_term_memory = []
        self.long_term_memory = []
        
        self.embedding_dim = 768
        
    def store(
        self,
        content: str,
        embedding: np.ndarray,
        emotional_state: np.ndarray,
        importance: Optional[float] = None
    ) -> str:
        
        if embedding.shape[0] != self.embedding_dim:
            embedding = self._normalize_embedding(embedding)
        
        emotional_tags = self._extract_emotional_tags(emotional_state)
        
        if importance is None:
            importance = self._calculate_importance(content, emotional_state)
        
        memory = MemoryTrace(
            content=content,
            embedding=embedding,
            emotional_tags=emotional_tags,
            importance=importance,
            access_count=1,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            memory_type='episodic'
        )
        
        self.working_memory.append(memory)
        self.short_term_memory.append(memory)
        
        if len(self.short_term_memory) > self.short_term_capacity:
            self._consolidate()
        
        return id(memory)
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[MemoryTrace]:
        
        if query_embedding.shape[0] != self.embedding_dim:
            query_embedding = self._normalize_embedding(query_embedding)
        
        all_memories = self.short_term_memory + self.long_term_memory
        
        if not all_memories:
            return []
        
        scores = []
        for memory in all_memories:
            similarity = np.dot(query_embedding, memory.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-8
            )
            
            recency = self._temporal_weight(memory)
            importance = memory.importance
            
            score = 0.4 * similarity + 0.3 * importance + 0.3 * recency
            scores.append(score)
            
            memory.access_count += 1
            memory.last_accessed = datetime.now()
        
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        return [all_memories[i] for i in sorted_indices]
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        if len(embedding) < self.embedding_dim:
            return np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        return embedding[:self.embedding_dim]
    
    def _extract_emotional_tags(self, emotional_state: np.ndarray) -> List[str]:
        emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise']
        tags = []
        for i, emotion in enumerate(emotion_labels):
            if i < len(emotional_state) and emotional_state[i] > 0.6:
                tags.append(emotion)
        return tags if tags else ['neutral']
    
    def _calculate_importance(self, content: str, emotional_state: np.ndarray) -> float:
        length_factor = min(len(content) / 1000.0, 1.0)
        question_factor = content.count('?') * 0.1
        emotional_factor = np.mean(emotional_state) if len(emotional_state) > 0 else 0.5
        
        importance = (length_factor + question_factor + emotional_factor) / 3.0
        return min(importance, 1.0)
    
    def _temporal_weight(self, memory: MemoryTrace) -> float:
        time_diff = (datetime.now() - memory.last_accessed).total_seconds()
        days = time_diff / 86400.0
        return np.exp(-days / 30.0)
    
    def _consolidate(self):
        candidates = sorted(
            self.short_term_memory,
            key=lambda m: m.importance,
            reverse=True
        )
        
        to_consolidate = [
            m for m in candidates 
            if m.importance > self.consolidation_threshold
        ][:len(candidates) // 2]
        
        for memory in to_consolidate:
            self.long_term_memory.append(memory)
            self.short_term_memory.remove(memory)
        
        if len(self.long_term_memory) > self.long_term_capacity:
            self.long_term_memory.sort(key=lambda m: m.importance)
            excess = len(self.long_term_memory) - self.long_term_capacity
            self.long_term_memory = self.long_term_memory[excess:]