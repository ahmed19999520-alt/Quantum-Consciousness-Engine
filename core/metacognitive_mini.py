import numpy as np
from typing import Dict, List, Any
from collections import deque

class MetacognitiveMonitor:
    def __init__(self):
        self.self_model = {
            'capabilities': [
                'language_processing',
                'pattern_recognition',
                'reasoning',
                'memory_retrieval'
            ],
            'limitations': [
                'no_physical_embodiment',
                'training_data_cutoff',
                'computational_constraints'
            ],
            'confidence_levels': {
                'language': 0.9,
                'reasoning': 0.8,
                'creativity': 0.7,
                'emotional_understanding': 0.6
            }
        }
        
        self.prediction_history = deque(maxlen=100)
        self.confidence_calibration = 0.5
        
    def assess_confidence(
        self,
        task_type: str,
        input_complexity: float,
        context_availability: float
    ) -> float:
        
        base_confidence = self.self_model['confidence_levels'].get(task_type, 0.5)
        
        complexity_factor = 1.0 - (input_complexity * 0.3)
        context_factor = context_availability * 0.2
        history_factor = self._compute_historical_accuracy() * 0.3
        
        confidence = base_confidence * (
            0.5 + 0.5 * (complexity_factor + context_factor + history_factor)
        )
        
        return np.clip(confidence, 0.0, 1.0)
    
    def detect_uncertainty(
        self,
        neural_outputs: np.ndarray,
        quantum_state: Any
    ) -> List[str]:
        
        uncertainty_sources = []
        
        output_variance = np.var(neural_outputs)
        if output_variance > 0.5:
            uncertainty_sources.append('high_output_variance')
        
        if hasattr(quantum_state, 'coherence') and quantum_state.coherence < 0.4:
            uncertainty_sources.append('low_quantum_coherence')
        
        if len(self.prediction_history) > 10:
            recent_accuracy = np.mean([p['correct'] for p in list(self.prediction_history)[-10:]])
            if recent_accuracy < 0.6:
                uncertainty_sources.append('recent_poor_performance')
        
        return uncertainty_sources
    
    def record_prediction(self, predicted: Any, actual: Any, correct: bool):
        self.prediction_history.append({
            'predicted': predicted,
            'actual': actual,
            'correct': correct,
            'timestamp': np.datetime64('now')
        })
        
        self._update_calibration()
    
    def _compute_historical_accuracy(self) -> float:
        if not self.prediction_history:
            return 0.5
        
        correct_count = sum(1 for p in self.prediction_history if p['correct'])
        return correct_count / len(self.prediction_history)
    
    def _update_calibration(self):
        if len(self.prediction_history) < 20:
            return
        
        recent = list(self.prediction_history)[-20:]
        accuracy = sum(1 for p in recent if p['correct']) / len(recent)
        
        self.confidence_calibration = 0.9 * self.confidence_calibration + 0.1 * accuracy