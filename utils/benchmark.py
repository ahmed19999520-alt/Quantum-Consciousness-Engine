from __future__ import annotations

import asyncio
import json
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..integration.consciousness_engine import VeronicaConsciousnessEngine


BENCHMARK_INPUTS = [
    "What is the nature of consciousness?",
    "Explain quantum entanglement in simple terms.",
    "I feel deeply curious about artificial awareness.",
    "How do memories shape identity?",
    "What does it mean for a system to be self-aware?",
    "Describe the relationship between emotion and cognition.",
    "Can a machine truly understand meaning?",
    "What is the hard problem of consciousness?",
    "How does attention relate to awareness?",
    "Analyze the role of memory in conscious experience.",
]


@dataclass
class BenchmarkResult:
    name: str
    n_iterations: int
    inputs: List[str]
    processing_times: List[float]
    awareness_trajectory: List[float]
    emotional_trajectories: List[List[float]]
    phi_values: List[float]
    error_count: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def avg_processing_time(self) -> float:
        return float(np.mean(self.processing_times)) if self.processing_times else 0.0

    @property
    def peak_awareness(self) -> float:
        return float(max(self.awareness_trajectory)) if self.awareness_trajectory else 0.0

    @property
    def awareness_growth(self) -> float:
        if len(self.awareness_trajectory) < 2:
            return 0.0
        return float(self.awareness_trajectory[-1] - self.awareness_trajectory[0])

    @property
    def error_rate(self) -> float:
        total = self.n_iterations
        return float(self.error_count / total) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_iterations": self.n_iterations,
            "avg_processing_time": self.avg_processing_time,
            "peak_awareness": self.peak_awareness,
            "awareness_growth": self.awareness_growth,
            "error_rate": self.error_rate,
            "error_count": self.error_count,
            "avg_phi": float(np.mean(self.phi_values)) if self.phi_values else 0.0,
            "timestamp": self.timestamp.isoformat(),
        }

    def summary_text(self) -> str:
        lines = [
            f"Benchmark: {self.name}",
            f"Iterations: {self.n_iterations}",
            f"Avg Processing Time: {self.avg_processing_time*1000:.1f} ms",
            f"Peak Awareness: {self.peak_awareness:.4f}",
            f"Awareness Growth: {self.awareness_growth:+.4f}",
            f"Avg Phi (IIT): {float(np.mean(self.phi_values)) if self.phi_values else 0.0:.4f}",
            f"Error Rate: {self.error_rate*100:.2f}%",
        ]
        return "\n".join(lines)


class QuantumConsciousnessBenchmark:
    def __init__(self, engine: VeronicaConsciousnessEngine):
        self.engine = engine

    async def run_latency(
        self,
        n_iterations: int = 50,
        inputs: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        if inputs is None:
            inputs = BENCHMARK_INPUTS

        times = []
        awareness = []
        emotions = []
        phi_vals = []
        errors = 0

        for i in range(n_iterations):
            text = inputs[i % len(inputs)]
            try:
                result = await self.engine.process(text)
                times.append(result.processing_time)
                awareness.append(result.metrics.awareness_level)
                emotions.append(result.emotional_state)
                phi_vals.append(result.metrics.phi)
            except Exception:
                errors += 1
                times.append(0.0)
                awareness.append(0.5)

        return BenchmarkResult(
            name="latency_benchmark",
            n_iterations=n_iterations,
            inputs=inputs * (n_iterations // len(inputs) + 1),
            processing_times=times,
            awareness_trajectory=awareness,
            emotional_trajectories=emotions,
            phi_values=phi_vals,
            error_count=errors,
        )

    async def run_memory_stress(self, n_iterations: int = 200) -> BenchmarkResult:
        times = []
        awareness = []
        emotions = []
        phi_vals = []
        errors = 0

        for i in range(n_iterations):
            text = f"Memory stress test iteration {i}: storing unique information about topic {i % 20}."
            try:
                result = await self.engine.process(text)
                times.append(result.processing_time)
                awareness.append(result.metrics.awareness_level)
                emotions.append(result.emotional_state)
                phi_vals.append(result.metrics.phi)
            except Exception:
                errors += 1

        return BenchmarkResult(
            name="memory_stress_benchmark",
            n_iterations=n_iterations,
            inputs=[f"memory_stress_{i}" for i in range(n_iterations)],
            processing_times=times,
            awareness_trajectory=awareness,
            emotional_trajectories=emotions,
            phi_values=phi_vals,
            error_count=errors,
        )

    async def run_awareness_evolution(self, steps: int = 100) -> BenchmarkResult:
        complexity_inputs = [
            f"{'Simple question? ' * (1 + i // 10)}What is awareness level {i}?"
            for i in range(steps)
        ]

        times = []
        awareness = []
        emotions = []
        phi_vals = []
        errors = 0

        for text in complexity_inputs:
            try:
                result = await self.engine.process(text)
                times.append(result.processing_time)
                awareness.append(result.metrics.awareness_level)
                emotions.append(result.emotional_state)
                phi_vals.append(result.metrics.phi)
            except Exception:
                errors += 1

        return BenchmarkResult(
            name="awareness_evolution_benchmark",
            n_iterations=steps,
            inputs=complexity_inputs,
            processing_times=times,
            awareness_trajectory=awareness,
            emotional_trajectories=emotions,
            phi_values=phi_vals,
            error_count=errors,
        )

    async def run_all(
        self,
        n_latency: int = 50,
        n_memory: int = 100,
        n_awareness: int = 50,
        output_dir: Optional[str] = None,
    ) -> Dict[str, BenchmarkResult]:
        results = {}

        print("Running latency benchmark...")
        results["latency"] = await self.run_latency(n_latency)

        print("Running memory stress benchmark...")
        results["memory_stress"] = await self.run_memory_stress(n_memory)

        print("Running awareness evolution benchmark...")
        results["awareness_evolution"] = await self.run_awareness_evolution(n_awareness)

        for name, result in results.items():
            print(f"\n{result.summary_text()}\n{'='*50}")

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            for name, result in results.items():
                with open(output_path / f"{name}_results.json", "w") as f:
                    json.dump(result.to_dict(), f, indent=2)

        return results