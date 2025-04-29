import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import Tuple, Type

from base_model import BaseRNN


class SeqFirstRNN(BaseRNN):
    def __init__(self, hidden_dim: int, key_dim: int, value_dim: int, output_dim: int, apply_contiguous=False):
        super().__init__(hidden_dim, key_dim, value_dim, output_dim)
        self.apply_contiguous = apply_contiguous

        self.query_proj = nn.Linear(hidden_dim, value_dim)         # query projection
        self.key_proj = nn.Linear(hidden_dim, key_dim)              # key projection
        self.value_proj = nn.Linear(hidden_dim, value_dim)          # value projection
        self.gate_proj = nn.Linear(hidden_dim, key_dim)             # gate projection
        self.out_proj = nn.Linear(value_dim, output_dim)            # output projection
    
    def _qkvg_proj(self, hidden_state):
        query = rearrange(self.query_proj(hidden_state), 'b t d -> t b d')           # (T, B, value_dim)
        key = rearrange(F.sigmoid(self.key_proj(hidden_state)), 'b t d -> t b d')    # (T, B, key_dim)
        value = rearrange(self.value_proj(hidden_state), 'b t d -> t b d')           # (T, B, value_dim)
        gate = rearrange(F.sigmoid(self.gate_proj(hidden_state)), 'b t d -> t b d')  # (T, B, key_dim)

        if self.apply_contiguous:
            query, key, value, gate = query.contiguous(), key.contiguous(), value.contiguous(), gate.contiguous()
        
        return query, key, value, gate
        
    def _rnn_loop(self, query, key, value, gate, B, T, dtype, device):
        K, V, O = self.key_dim, self.value_dim, self.output_dim

        state = torch.zeros(B, K, V, dtype=dtype, device=device)
        output = torch.zeros(T, B, O, dtype=dtype, device=device)  # !Sequence length first

        for i in range(T):
            query_i = query[i]                    # (B, value_dim)
            key_i = key[i]                        # (B, key_dim)
            value_i = value[i]                    # (B, value_dim)
            gate_i = gate[i]                      # (B, key_dim)

            key_value_i = key_i.unsqueeze(-1) * value_i.unsqueeze(1)  # (B, key_dim, value_dim)
            state = state * gate_i.unsqueeze(-1) + key_value_i        # (B, key_dim, value_dim)
            output[i] = self.out_proj((query_i.unsqueeze(-1) * state).sum(-2))       # (B, output_dim)
        
        output = rearrange(output, 't b d -> b t d')
        
        if self.apply_contiguous:
            output = output.contiguous()
        
        return output, state
