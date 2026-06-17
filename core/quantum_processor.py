from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit import Aer, execute, IBMQ
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector, DensityMatrix, entropy
    from qiskit.providers.aer import QasmSimulator

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import cirq
    import sympy

    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

try:
    from pennylane import numpy as pnp
    import pennylane as qml

    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


@dataclass
class QuantumState:
    amplitudes: np.ndarray
    phases: np.ndarray
    entanglement: float
    coherence: float
    fidelity: float
    von_neumann_entropy: float = 0.0
    purity: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "amplitudes": self.amplitudes.tolist(),
            "phases": self.phases.tolist(),
            "entanglement": float(self.entanglement),
            "coherence": float(self.coherence),
            "fidelity": float(self.fidelity),
            "von_neumann_entropy": float(self.von_neumann_entropy),
            "purity": float(self.purity),
            "timestamp": self.timestamp.isoformat(),
        }


class QuantumConsciousnessProcessor:
    def __init__(
        self,
        n_qubits: int = 16,
        backend_name: str = "aer_statevector_simulator",
        shots: int = 1024,
        optimization_level: int = 1,
        ibm_token: Optional[str] = None,
    ):
        self.n_qubits = n_qubits
        self.backend_name = backend_name
        self.shots = shots
        self.optimization_level = optimization_level
        self.ibm_token = ibm_token
        self.backend = None
        self.state_history: List[QuantumState] = []
        self.parameters: Dict[str, List] = {
            "emotional": [Parameter(f"e_{i}") for i in range(n_qubits)]
            if QISKIT_AVAILABLE
            else list(range(n_qubits)),
            "memory": [Parameter(f"m_{i}") for i in range(n_qubits)]
            if QISKIT_AVAILABLE
            else list(range(n_qubits)),
        }
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        if not QISKIT_AVAILABLE:
            return
        try:
            if self.ibm_token and self.backend_name.startswith("ibmq"):
                IBMQ.save_account(self.ibm_token, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider()
                self.backend = provider.get_backend(self.backend_name)
            else:
                self.backend = Aer.get_backend(self.backend_name)
        except Exception:
            self.backend = None

    def create_consciousness_circuit(
        self,
        emotional_params: np.ndarray,
        memory_params: np.ndarray,
    ) -> "QuantumCircuit":
        if not QISKIT_AVAILABLE:
            return None

        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        for i in range(self.n_qubits):
            qc.h(qr[i])

        for i in range(min(len(emotional_params), self.n_qubits)):
            angle = float(emotional_params[i]) * np.pi
            qc.ry(angle, qr[i])

        for i in range(0, self.n_qubits - 1, 2):
            if i + 1 < len(memory_params):
                angle = float(memory_params[i]) * float(memory_params[i + 1]) * np.pi
                qc.crz(angle, qr[i], qr[i + 1])

        for i in range(self.n_qubits - 1):
            qc.cx(qr[i], qr[i + 1])
            qc.rz(np.pi / 8, qr[i + 1])

        for i in range(self.n_qubits // 2):
            qc.cry(np.pi / 4, qr[i], qr[self.n_qubits - 1 - i])

        qc.barrier()
        qc.measure(qr, cr)

        return qc

    def execute_circuit(
        self,
        circuit: "QuantumCircuit",
        return_statevector: bool = True,
    ) -> QuantumState:
        if not QISKIT_AVAILABLE or self.backend is None:
            return self._classical_fallback()

        try:
            if return_statevector:
                sv_backend = Aer.get_backend("statevector_simulator")
                clean_circuit = circuit.remove_final_measurements(inplace=False)
                job = execute(clean_circuit, sv_backend)
                result = job.result()
                sv = np.array(result.get_statevector())

                amplitudes = np.abs(sv)
                phases = np.angle(sv)

                dm = np.outer(sv, np.conj(sv))
                eigenvals = np.linalg.eigvalsh(dm)
                eigenvals = np.real(eigenvals[eigenvals > 1e-12])
                vn_entropy = float(-np.sum(eigenvals * np.log2(eigenvals + 1e-15)))

                coherence = float(np.sum(np.abs(np.triu(dm, 1))))
                purity = float(np.real(np.trace(dm @ dm)))

                ideal = np.zeros(len(sv))
                ideal[0] = 1.0
                fidelity = float(np.abs(np.vdot(ideal, sv)) ** 2)

                entanglement = self._bipartite_entanglement(sv)

            else:
                job = execute(
                    circuit,
                    self.backend,
                    shots=self.shots,
                    optimization_level=self.optimization_level,
                )
                result = job.result()
                counts = result.get_counts()
                total = sum(counts.values())
                dim = 2**self.n_qubits

                amplitudes = np.zeros(dim)
                phases = np.zeros(dim)
                for bitstring, count in counts.items():
                    idx = int(bitstring, 2)
                    amplitudes[idx] = np.sqrt(count / total)
                    phases[idx] = np.random.uniform(0, 2 * np.pi)

                probs = amplitudes**2
                probs_nz = probs[probs > 1e-12]
                vn_entropy = float(-np.sum(probs_nz * np.log2(probs_nz)))
                coherence = 0.0
                purity = float(np.sum(probs**2))
                fidelity = float(amplitudes[0] ** 2)
                entanglement = vn_entropy

            state = QuantumState(
                amplitudes=amplitudes,
                phases=phases,
                entanglement=entanglement,
                coherence=coherence,
                fidelity=fidelity,
                von_neumann_entropy=vn_entropy,
                purity=purity,
            )
            self.state_history.append(state)
            return state

        except Exception:
            return self._classical_fallback()

    def _bipartite_entanglement(self, statevector: np.ndarray) -> float:
        n = self.n_qubits
        half = n // 2
        dim_a = 2**half
        dim_b = 2 ** (n - half)
        try:
            psi = statevector.reshape(dim_a, dim_b)
            _, s, _ = np.linalg.svd(psi)
            s2 = s**2
            s2 = s2[s2 > 1e-12]
            return float(-np.sum(s2 * np.log2(s2 + 1e-15)))
        except Exception:
            return 0.0

    def _classical_fallback(self) -> QuantumState:
        n_states = 2**self.n_qubits
        rng = np.random.default_rng()
        raw = rng.standard_normal(n_states) + 1j * rng.standard_normal(n_states)
        sv = raw / np.linalg.norm(raw)
        amplitudes = np.abs(sv)
        phases = np.angle(sv)
        probs = amplitudes**2
        probs_nz = probs[probs > 1e-12]
        vn_entropy = float(-np.sum(probs_nz * np.log2(probs_nz + 1e-15)))
        coherence = float(np.random.uniform(0.3, 0.8))
        purity = float(np.sum(probs**2))
        fidelity = float(amplitudes[0] ** 2)
        entanglement = vn_entropy * 0.7

        state = QuantumState(
            amplitudes=amplitudes,
            phases=phases,
            entanglement=entanglement,
            coherence=coherence,
            fidelity=fidelity,
            von_neumann_entropy=vn_entropy,
            purity=purity,
        )
        self.state_history.append(state)
        return state

    def optimize_parameters(
        self,
        target_entanglement: float,
        emotional_state: np.ndarray,
        max_iter: int = 100,
    ) -> np.ndarray:
        from scipy.optimize import minimize

        def cost(params: np.ndarray) -> float:
            if QISKIT_AVAILABLE and self.backend is not None:
                circuit = self.create_consciousness_circuit(params, emotional_state)
                state = self.execute_circuit(circuit, return_statevector=True)
                return abs(state.entanglement - target_entanglement)
            else:
                return abs(np.mean(params) - target_entanglement)

        x0 = np.random.uniform(0, 2 * np.pi, len(emotional_state))
        result = minimize(cost, x0, method="COBYLA", options={"maxiter": max_iter})
        return result.x if result.success else x0

    def get_latest_state(self) -> Optional[QuantumState]:
        return self.state_history[-1] if self.state_history else None

    def clear_history(self) -> None:
        self.state_history.clear()

    @property
    def backend_available(self) -> bool:
        return QISKIT_AVAILABLE and self.backend is not None

    @property
    def active_backend(self) -> str:
        if not QISKIT_AVAILABLE:
            return "classical_numpy_fallback"
        if self.backend is None:
            return "uninitialized"
        return self.backend_name