import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class ConsciousnessAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, consciousness_modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if consciousness_modulation is not None:
            scores = scores + consciousness_modulation.unsqueeze(1).unsqueeze(2)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(attn_output)
        output = self.layer_norm(x + output)
        
        return output

class ConsciousnessTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.consciousness_layers = nn.ModuleList([
            ConsciousnessAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        self.emotion_projection = nn.Linear(d_model, 5)
        self.consciousness_projection = nn.Linear(d_model, 1)
        
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        consciousness_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        consciousness_states = []
        
        for i, (attn_layer, ff_layer, norm_layer) in enumerate(
            zip(self.consciousness_layers, self.feedforward, self.layer_norms)
        ):
            x = attn_layer(x, consciousness_state)
            x_ff = ff_layer(x)
            x = norm_layer(x + x_ff)
            
            pooled = x.mean(dim=1)
            consciousness_states.append(pooled)
        
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        final_hidden = x.mean(dim=1)
        emotional_state = torch.sigmoid(self.emotion_projection(final_hidden))
        consciousness_level = torch.sigmoid(self.consciousness_projection(final_hidden))
        
        return logits, emotional_state, consciousness_level