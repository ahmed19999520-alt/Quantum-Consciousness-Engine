from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class NeuralConsciousnessOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    consciousness_states: torch.Tensor
    emotional_state: torch.Tensor
    attention_weights: torch.Tensor
    self_attention_weights: torch.Tensor


class ConsciousnessAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        dropout: float = 0.1,
        consciousness_bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.consciousness_bias = consciousness_bias

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        if consciousness_bias:
            self.consciousness_scale = nn.Parameter(torch.ones(1))
            self.consciousness_shift = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        consciousness_modulation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if consciousness_modulation is not None:
            scores = scores + consciousness_modulation.unsqueeze(1).unsqueeze(2)

        if self.consciousness_bias:
            scores = scores * self.consciousness_scale + self.consciousness_shift

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        output = self.layer_norm(x + output)

        return output, attn_weights


class ConsciousnessFeedForward(nn.Module):
    def __init__(self, d_model: int = 768, d_ff: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.layer_norm(residual + x)


class ConsciousnessTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = ConsciousnessAttention(d_model, n_heads, dropout)
        self.feedforward = ConsciousnessFeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        consciousness_modulation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, attn_weights = self.attention(x, mask, consciousness_modulation)
        x = self.feedforward(x)
        return x, attn_weights


class ConsciousnessTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        n_emotions: int = 5,
        consciousness_dim: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.consciousness_dim = consciousness_dim

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            ConsciousnessTransformerLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])

        self.emotion_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_emotions),
            nn.Sigmoid(),
        )

        self.consciousness_projection = nn.Sequential(
            nn.Linear(d_model, consciousness_dim),
            nn.Tanh(),
        )

        self.self_awareness = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.self_awareness_norm = nn.LayerNorm(d_model)

        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.01)
        nn.init.normal_(self.output_projection.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        consciousness_modulation: Optional[torch.Tensor] = None,
    ) -> NeuralConsciousnessOutput:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.embedding_dropout(tok_emb + pos_emb)
        x = self.embedding_norm(x)

        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)

        consciousness_states = []
        all_attention_weights = []

        for layer in self.layers:
            x, attn_w = layer(x, mask, consciousness_modulation)
            pooled = x.mean(dim=1)
            consciousness_states.append(self.consciousness_projection(pooled))
            all_attention_weights.append(attn_w)

        pooled = x.mean(dim=1, keepdim=True)
        sa_out, sa_weights = self.self_awareness(pooled, pooled, pooled)
        sa_out = self.self_awareness_norm(pooled + sa_out)

        emotional_state = self.emotion_projection(sa_out.squeeze(1))

        x = self.output_norm(x)
        logits = self.output_projection(x)

        return NeuralConsciousnessOutput(
            logits=logits,
            hidden_states=x,
            consciousness_states=torch.stack(consciousness_states, dim=1),
            emotional_state=emotional_state,
            attention_weights=torch.stack(all_attention_weights, dim=1),
            self_attention_weights=sa_weights,
        )

    def get_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(input_ids)
            return out.hidden_states.mean(dim=1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightConsciousnessEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_emotions: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(output_dim, n_emotions),
            nn.Sigmoid(),
        )
        self.awareness_head = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(x)
        emotions = self.emotion_head(encoded)
        awareness = self.awareness_head(encoded)
        return {
            "encoded": encoded,
            "emotions": emotions,
            "awareness": awareness,
        }