import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class IBMQuantumIntegration:
    
    def __init__(self, api_token: str):
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=api_token, overwrite=True)
            self.service = QiskitRuntimeService()
        except ImportError:
            raise ImportError("qiskit-ibm-runtime not installed")
    
    def get_available_backends(self) -> List[Dict[str, Any]]:
        try:
            backends = self.service.backends(simulator=False, operational=True)
            
            result = []
            for backend in backends:
                result.append({
                    'name': backend.name,
                    'qubits': backend.num_qubits,
                    'queue_depth': backend.queue_depth,
                    'operational': backend.operational,
                    'basis_gates': backend.basis_gates
                })
            
            return result
        except Exception as e:
            logger.error(f"Failed to get backends: {e}")
            return []
    
    def submit_job(self, circuit: Any, backend_name: str, shots: int = 1024) -> str:
        try:
            backend = self.service.backend(backend_name)
            
            from qiskit_ibm_runtime import Session
            
            with Session(backend=backend) as session:
                job = session.run(circuit, shots=shots)
                return job.job_id
        except Exception as e:
            logger.error(f"Job submission failed: {e}")
            return None
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        try:
            job = self.service.job(job_id)
            
            return {
                'job_id': job_id,
                'status': job.status,
                'backend': job.backend_name,
                'creation_date': str(job.creation_date),
                'queue_position': getattr(job, 'queue_position', None)
            }
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {'error': str(e)}
    
    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        try:
            job = self.service.job(job_id)
            
            if job.status != 'DONE':
                return {'status': job.status, 'ready': False}
            
            result = job.result()
            counts = result.get_counts()
            
            return {
                'job_id': job_id,
                'counts': counts,
                'backend': job.backend_name,
                'ready': True,
                'execution_time': result.time_taken
            }
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
            return {'error': str(e)}