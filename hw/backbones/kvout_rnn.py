import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import Tuple, Type

from base_model import BaseRNN


class KVOutRNN(BaseRNN):
    def __init__(self, hidden_dim: int, key_dim: int, value_dim: int, output_dim: int, apply_contiguous=False):
        super().__init__(hidden_dim, key_dim, value_dim, output_dim)
        self.apply_contiguous = apply_contiguous
        self.qkvg_dims = [value_dim, key_dim, value_dim, key_dim] 
        self.qkvg_proj = nn.Linear(hidden_dim, sum(self.qkvg_dims))  # query, key, value, gate fused projection
        self.out_proj = nn.Linear(value_dim, output_dim)             # output projection
    
    def _qkvg_proj(self, hidden_state):
        qkvg = self.qkvg_proj(hidden_state)        # (B, T, sum(self.qkvg_dims))
        qkvg = rearrange(qkvg, 'b t d -> t b d')   # (T, B, sum(self.qkvg_dims))
        
        if self.apply_contiguous:
            qkvg = qkvg.contiguous()
        
        return qkvg.split(self.qkvg_dims, dim=-1)  # split into 4 tensors according to self.qkvg_dims
        
    def _rnn_loop(self, query, key, value, gate, B, T, dtype, device):
        # Same function as SeqFirstRNN one, copied for more readability
        K, V, O = self.key_dim, self.value_dim, self.output_dim

        state = torch.zeros(B, K, V, dtype=dtype, device=device)
        output = torch.zeros(T, B, O, dtype=dtype, device=device)  # !Sequence length first

        key_value = key.unsqueeze(-1) * value.unsqueeze(-2)  # (T, B, key_dim, value_dim)

        for i in range(T):
            query_i = query[i]                    # (B, value_dim)
            key_value_i = key_value[i]                        # (B, key_dim)
            gate_i = gate[i]                      # (B, key_dim)

            state = state * gate_i.unsqueeze(-1) + key_value_i        # (B, key_dim, value_dim)
            output[i] = self.out_proj((query_i.unsqueeze(-1) * state).sum(-2))       # (B, output_dim)
        
        output = rearrange(output, 't b d -> b t d')
        
        if self.apply_contiguous:
            output = output.contiguous()
        
        return output, state
