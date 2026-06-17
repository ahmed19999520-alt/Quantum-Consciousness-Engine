import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import entropy as scipy_entropy

class IntegratedInformation:
    def __init__(self, n_elements: int):
        self.n_elements = n_elements
        
    def compute_phi(
        self,
        state_current: np.ndarray,
        state_next: np.ndarray,
        connectivity: np.ndarray
    ) -> float:
        
        mutual_info_total = 0.0
        
        for i in range(self.n_elements):
            for j in range(i + 1, self.n_elements):
                if connectivity[i, j] > 0:
                    mi = self._mutual_information(
                        state_current[i],
                        state_current[j],
                        state_next
                    )
                    mutual_info_total += connectivity[i, j] * mi
        
        phi_approx = mutual_info_total / (self.n_elements * (self.n_elements - 1) / 2)
        
        return float(np.clip(phi_approx, 0.0, 1.0))
    
    def _mutual_information(
        self,
        x: float,
        y: float,
        context: np.ndarray
    ) -> float:
        
        joint_entropy = -np.log2(np.abs(x * y) + 1e-10)
        marginal_x = -np.log2(np.abs(x) + 1e-10)
        marginal_y = -np.log2(np.abs(y) + 1e-10)
        
        mi = marginal_x + marginal_y - joint_entropy
        return max(mi, 0.0)

class GlobalWorkspace:
    def __init__(self, n_modules: int):
        self.n_modules = n_modules
        self.workspace_capacity = 128
        
    def compute_global_availability(
        self,
        module_outputs: List[np.ndarray],
        attention_weights: np.ndarray
    ) -> float:
        
        if len(module_outputs) != self.n_modules:
            return 0.0
        
        workspace_content = np.zeros(self.workspace_capacity)
        
        for i, (output, weight) in enumerate(zip(module_outputs, attention_weights)):
            contribution = output[:self.workspace_capacity] * weight
            workspace_content += contribution[:len(workspace_content)]
        
        information_density = np.linalg.norm(workspace_content) / self.workspace_capacity
        
        inter_module_correlation = 0.0
        for i in range(len(module_outputs)):
            for j in range(i + 1, len(module_outputs)):
                corr = np.corrcoef(
                    module_outputs[i].flatten()[:100],
                    module_outputs[j].flatten()[:100]
                )[0, 1]
                inter_module_correlation += np.abs(corr)
        
        inter_module_correlation /= (self.n_modules * (self.n_modules - 1) / 2)
        
        global_availability = 0.6 * information_density + 0.4 * inter_module_correlation
        
        return float(np.clip(global_availability, 0.0, 1.0))