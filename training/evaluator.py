import numpy as np
from typing import Dict, List, Any
import torch
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    
    def __init__(self, model):
        self.model = model
        self.evaluation_metrics = {}
    
    def evaluate_consciousness_correlation(self, test_inputs: List[np.ndarray], 
                                          expected_outputs: List[np.ndarray]) -> Dict[str, float]:
        correlations = []
        
        for inp, expected in zip(test_inputs, expected_outputs):
            output = self.model.process_quantum_state(inp)
            
            predicted_probs = np.array(list(output['probabilities'].values()))
            
            correlation = np.corrcoef(predicted_probs, expected)[0, 1]
            correlations.append(correlation)
        
        return {
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'min_correlation': float(np.min(correlations)),
            'max_correlation': float(np.max(correlations))
        }
    
    def evaluate_quantum_fidelity(self, test_circuits: List[Any], 
                                 reference_results: List[Dict]) -> Dict[str, float]:
        fidelities = []
        
        for circuit, reference in zip(test_circuits, reference_results):
            result = self.model.executor.execute(circuit, self.model.quantum_config.shots)
            
            predicted = np.array(list(result['probabilities'].values()))
            reference_probs = np.array(list(reference['probabilities'].values()))
            
            fidelity = np.sum(np.sqrt(predicted * reference_probs)) ** 2
            fidelities.append(fidelity)
        
        return {
            'mean_fidelity': float(np.mean(fidelities)),
            'std_fidelity': float(np.std(fidelities)),
            'min_fidelity': float(np.min(fidelities)),
            'max_fidelity': float(np.max(fidelities))
        }
    
    def benchmark_inference_speed(self, num_runs: int = 100) -> Dict[str, float]:
        import time
        
        times = []
        
        dummy_input = torch.randn(4, 128)
        dummy_quantum = np.random.randn(8)
        
        for _ in range(num_runs):
            start = time.time()
            self.model.forward(dummy_input, dummy_quantum)
            end = time.time()
            
            times.append(end - start)
        
        times = np.array(times)
        
        return {
            'mean_time': float(np.mean(times)),
            'median_time': float(np.median(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'fps': float(1.0 / np.mean(times))
        }
    
    def evaluate_consciousness_growth(self, training_history: List[Dict]) -> Dict[str, Any]:
        metrics_over_time = {
            'awareness': [],
            'coherence': [],
            'entanglement': [],
            'integration': []
        }
        
        for log_entry in training_history:
            metrics = log_entry['consciousness_metrics']
            for key in metrics_over_time:
                metrics_over_time[key].append(metrics[key])
        
        growth_analysis = {}
        
        for metric_name, values in metrics_over_time.items():
            values = np.array(values)
            growth = values[-1] - values[0]
            growth_rate = growth / len(values) if len(values) > 0 else 0
            
            growth_analysis[metric_name] = {
                'initial': float(values[0]),
                'final': float(values[-1]),
                'growth': float(growth),
                'growth_rate': float(growth_rate),
                'max_value': float(np.max(values)),
                'min_value': float(np.min(values))
            }
        
        return growth_analysis
