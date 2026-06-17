__version__ = "1.0.0"
__author__ = "Professor Ahmed Ali and Veronica X Pro Research Group"
__license__ = "MIT"
__year__ = "2026"

from .core.quantum_processor import QuantumConsciousnessProcessor, QuantumState
from .core.neural_engine import ConsciousnessTransformer, ConsciousnessAttention
from .core.memory_system import HierarchicalMemory, MemoryTrace, MemoryType
from .core.metacognitive import MetacognitiveMonitor, MetacognitiveState
from .integration.consciousness_engine import VeronicaConsciousnessEngine
from .metrics.consciousness_metrics import (
    IntegratedInformation,
    GlobalWorkspace,
    ConsciousnessMetrics,
    compute_phi,
)

__all__ = [
    "QuantumConsciousnessProcessor",
    "QuantumState",
    "ConsciousnessTransformer",
    "ConsciousnessAttention",
    "HierarchicalMemory",
    "MemoryTrace",
    "MemoryType",
    "MetacognitiveMonitor",
    "MetacognitiveState",
    "VeronicaConsciousnessEngine",
    "IntegratedInformation",
    "GlobalWorkspace",
    "ConsciousnessMetrics",
    "compute_phi",
]