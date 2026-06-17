from __future__ import annotations

import json
import os
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, Dataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.neural_engine import ConsciousnessTransformer, LightweightConsciousnessEncoder
from ..core.memory_system import HierarchicalMemory, MemoryType
from ..metrics.consciousness_metrics import IntegratedInformation


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    batch_size: int = 16
    n_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 100
    save_every: int = 500
    max_seq_len: int = 512
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    train_split: float = 0.9
    device: str = "cuda" if TORCH_AVAILABLE and __import__("torch").cuda.is_available() else "cpu"
    output_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    seed: int = 42


@dataclass
class TrainingStep:
    step: int
    epoch: int
    loss: float
    lr: float
    elapsed: float
    metrics: Dict[str, float]


class ConsciousnessDataset(Dataset if TORCH_AVAILABLE else object):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer_fn: Callable[[str], List[int]],
        max_seq_len: int = 512,
    ):
        self.data = data
        self.tokenizer_fn = tokenizer_fn
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        item = self.data[idx]
        text = item.get("text", "")
        tokens = self.tokenizer_fn(text)
        tokens = tokens[: self.max_seq_len]

        if TORCH_AVAILABLE:
            import torch
            input_ids = torch.tensor(tokens, dtype=torch.long)
            attn_mask = torch.ones_like(input_ids)
            emotional_target = torch.tensor(
                item.get("emotional_state", [0.5] * 5), dtype=torch.float32
            )
            awareness_target = torch.tensor(
                [item.get("awareness_level", 0.5)], dtype=torch.float32
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "emotional_target": emotional_target,
                "awareness_target": awareness_target,
            }
        return {}


class JsonlDataLoader:
    def __init__(self, path: str, max_samples: Optional[int] = None):
        self.path = Path(path)
        self.max_samples = max_samples

    def load(self) -> List[Dict[str, Any]]:
        records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                    if self.max_samples and len(records) >= self.max_samples:
                        break
                except json.JSONDecodeError:
                    continue
        return records


class SyntheticConsciousnessDataGenerator:
    CONVERSATION_TEMPLATES = [
        ("What is consciousness?", 0.8, [0.3, 0.7, 0.2, 0.1, 0.6]),
        ("I feel deeply sad today.", 0.6, [0.1, 0.9, 0.2, 0.4, 0.1]),
        ("Explain quantum entanglement.", 0.9, [0.4, 0.6, 0.3, 0.1, 0.8]),
        ("I am curious about the universe.", 0.7, [0.7, 0.3, 0.1, 0.1, 0.9]),
        ("Can you help me with this problem?", 0.65, [0.5, 0.3, 0.2, 0.3, 0.6]),
        ("I feel angry and frustrated.", 0.55, [0.1, 0.4, 0.9, 0.3, 0.2]),
        ("What does it mean to be aware?", 0.85, [0.4, 0.7, 0.2, 0.2, 0.7]),
        ("Tell me something creative.", 0.7, [0.8, 0.3, 0.2, 0.1, 0.8]),
        ("I need support right now.", 0.6, [0.2, 0.8, 0.1, 0.5, 0.2]),
        ("Analyze this philosophical argument.", 0.88, [0.3, 0.6, 0.2, 0.1, 0.7]),
    ]

    def generate(self, n_samples: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        samples = []
        for i in range(n_samples):
            template = self.CONVERSATION_TEMPLATES[i % len(self.CONVERSATION_TEMPLATES)]
            text, awareness, emotions = template
            noise = rng.normal(0, 0.05, 5)
            emotional_state = list(np.clip(np.array(emotions) + noise, 0.0, 1.0))
            awareness_noisy = float(np.clip(awareness + rng.normal(0, 0.03), 0.0, 1.0))
            samples.append({
                "text": text,
                "awareness_level": awareness_noisy,
                "emotional_state": emotional_state,
                "sample_id": i,
            })
        return samples

    def save_jsonl(self, path: str, n_samples: int = 1000, seed: int = 42) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        samples = self.generate(n_samples, seed)
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


class ConsciousnessTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        self._history: List[TrainingStep] = []
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    def build_model(self) -> Optional["nn.Module"]:
        if not TORCH_AVAILABLE:
            return None
        model = LightweightConsciousnessEncoder(
            input_dim=self.config.d_model,
            hidden_dim=self.config.d_model * 2,
            output_dim=self.config.d_model,
            n_emotions=5,
            dropout=self.config.dropout,
        )
        return model.to(self.device)

    def train(
        self,
        data: List[Dict[str, Any]],
        embedding_fn: Callable[[str], np.ndarray],
        model: Optional["nn.Module"] = None,
    ) -> List[TrainingStep]:
        if not TORCH_AVAILABLE:
            return self._dummy_training(data)

        if model is None:
            model = self.build_model()

        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion_emotion = nn.MSELoss()
        criterion_awareness = nn.MSELoss()

        np.random.seed(self.config.seed)
        np.random.shuffle(data)
        split = int(len(data) * self.config.train_split)
        train_data = data[:split]
        val_data = data[split:]

        global_step = 0
        for epoch in range(self.config.n_epochs):
            model.train()
            epoch_losses = []

            for batch_start in range(0, len(train_data), self.config.batch_size):
                batch = train_data[batch_start: batch_start + self.config.batch_size]
                if not batch:
                    continue

                t0 = time.perf_counter()

                embeddings = np.stack([embedding_fn(item["text"]) for item in batch])
                emotions = np.array([item.get("emotional_state", [0.5] * 5) for item in batch])
                awareness = np.array([[item.get("awareness_level", 0.5)] for item in batch])

                import torch
                x = torch.tensor(
                    embeddings[:, : self.config.d_model], dtype=torch.float32, device=self.device
                )
                emotion_targets = torch.tensor(emotions, dtype=torch.float32, device=self.device)
                awareness_targets = torch.tensor(awareness, dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                out = model(x)

                emotion_loss = criterion_emotion(out["emotions"], emotion_targets)
                awareness_loss = criterion_awareness(out["awareness"], awareness_targets)
                loss = emotion_loss + awareness_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

                step_loss = float(loss.item())
                epoch_losses.append(step_loss)
                elapsed = time.perf_counter() - t0

                step = TrainingStep(
                    step=global_step,
                    epoch=epoch,
                    loss=step_loss,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    elapsed=elapsed,
                    metrics={"emotion_loss": float(emotion_loss), "awareness_loss": float(awareness_loss)},
                )
                self._history.append(step)
                global_step += 1

                if global_step % self.config.eval_every == 0:
                    val_loss = self._evaluate(model, val_data, embedding_fn, criterion_emotion, criterion_awareness)
                    print(
                        f"Epoch {epoch+1} | Step {global_step} | Train Loss: {step_loss:.4f} | Val Loss: {val_loss:.4f}"
                    )

                if global_step % self.config.save_every == 0:
                    self._save_checkpoint(model, optimizer, epoch, global_step)

            print(f"Epoch {epoch+1}/{self.config.n_epochs} | Avg Loss: {np.mean(epoch_losses):.4f}")

        self._save_checkpoint(model, optimizer, self.config.n_epochs - 1, global_step, final=True)
        return self._history

    def _evaluate(
        self,
        model: "nn.Module",
        data: List[Dict[str, Any]],
        embedding_fn: Callable[[str], np.ndarray],
        criterion_emotion: "nn.Module",
        criterion_awareness: "nn.Module",
    ) -> float:
        if not data:
            return 0.0

        import torch
        model.eval()
        losses = []
        with torch.no_grad():
            for batch_start in range(0, len(data), self.config.batch_size):
                batch = data[batch_start: batch_start + self.config.batch_size]
                if not batch:
                    continue
                embeddings = np.stack([embedding_fn(item["text"]) for item in batch])
                emotions = np.array([item.get("emotional_state", [0.5] * 5) for item in batch])
                awareness = np.array([[item.get("awareness_level", 0.5)] for item in batch])

                x = torch.tensor(embeddings[:, : self.config.d_model], dtype=torch.float32, device=self.device)
                emotion_targets = torch.tensor(emotions, dtype=torch.float32, device=self.device)
                awareness_targets = torch.tensor(awareness, dtype=torch.float32, device=self.device)

                out = model(x)
                loss = criterion_emotion(out["emotions"], emotion_targets) + criterion_awareness(out["awareness"], awareness_targets)
                losses.append(float(loss.item()))

        model.train()
        return float(np.mean(losses)) if losses else 0.0

    def _save_checkpoint(
        self,
        model: "nn.Module",
        optimizer,
        epoch: int,
        step: int,
        final: bool = False,
    ) -> None:
        import torch
        tag = "final" if final else f"epoch{epoch}_step{step}"
        path = Path(self.config.output_dir) / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history_length": len(self._history),
            },
            path,
        )

    def _dummy_training(self, data: List[Dict[str, Any]]) -> List[TrainingStep]:
        print("PyTorch not available. Running dummy training simulation.")
        history = []
        for step in range(min(10, len(data))):
            history.append(TrainingStep(
                step=step,
                epoch=0,
                loss=float(np.random.uniform(0.1, 1.0)),
                lr=self.config.learning_rate,
                elapsed=0.01,
                metrics={},
            ))
        return history

    @property
    def training_history(self) -> List[TrainingStep]:
        return self._history