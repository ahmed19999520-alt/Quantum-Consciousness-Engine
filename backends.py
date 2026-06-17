import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class QuantumBackend(ABC):
    
    @abstractmethod
    def execute_circuit(self, circuit: Any, shots: int) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        pass


class QiskitBackend(QuantumBackend):
    
    def __init__(self):
        try:
            from qiskit import Aer
            self.aer = Aer
            self.backend = Aer.get_backend('qasm_simulator')
        except ImportError:
            raise ImportError("Qiskit not installed")
    
    def execute_circuit(self, circuit: Any, shots: int = 1024) -> Dict[str, Any]:
        try:
            from qiskit import execute
            job = execute(circuit, self.backend, shots=shots)
            result = job.result()
            counts = result.get_counts(circuit)
            
            return {
                'counts': counts,
                'backend': 'qiskit',
                'shots': shots,
                'successful': True
            }
        except Exception as e:
            logger.error(f"Qiskit execution failed: {e}")
            return {'successful': False, 'error': str(e)}
    
    def get_backend_info(self) -> Dict[str, Any]:
        return {
            'name': 'Qiskit',
            'version': '0.36.0+',
            'type': 'simulator',
            'supports': ['statevector', 'qasm', 'unitary']
        }


class CirqBackend(QuantumBackend):
    
    def __init__(self):
        try:
            import cirq
            self.cirq = cirq
        except ImportError:
            raise ImportError("Cirq not installed")
    
    def execute_circuit(self, circuit: Any, shots: int = 1024) -> Dict[str, Any]:
        try:
            simulator = self.cirq.Simulator()
            result = simulator.simulate(circuit, repetitions=shots)
            
            measurements = result.measurements
            
            return {
                'measurements': measurements,
                'backend': 'cirq',
                'shots': shots,
                'successful': True
            }
        except Exception as e:
            logger.error(f"Cirq execution failed: {e}")
            return {'successful': False, 'error': str(e)}
    
    def get_backend_info(self) -> Dict[str, Any]:
        return {
            'name': 'Cirq',
            'version': '0.14.0+',
            'type': 'simulator',
            'supports': ['clifford', 'stabilizer', 'full']
        }


class BraketBackend(QuantumBackend):
    
    def __init__(self, device_arn: Optional[str] = None):
        try:
            from braket.aws import AwsDevice
            self.braket = AwsDevice
            self.device_arn = device_arn or 'arn:aws:braket:::device/quantum-simulator/amazon/tn1'
        except ImportError:
            raise ImportError("Amazon Braket SDK not installed")
    
    def execute_circuit(self, circuit: Any, shots: int = 100) -> Dict[str, Any]:
        try:
            device = self.braket(self.device_arn)
            task = device.run(circuit, shots=shots)
            result = task.result()
            
            return {
                'result': result,
                'backend': 'braket',
                'shots': shots,
                'successful': True,
                'task_arn': task.id
            }
        except Exception as e:
            logger.error(f"Braket execution failed: {e}")
            return {'successful': False, 'error': str(e)}
    
    def get_backend_info(self) -> Dict[str, Any]:
        return {
            'name': 'Amazon Braket',
            'version': '1.0.0+',
            'type': 'cloud',
            'device_arn': self.device_arn
        }
