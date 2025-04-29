import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SimpleRNN(nn.Module):
    def __init__(self, hidden_dim: int, key_dim: int, value_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim

        self.query_proj = nn.Linear(hidden_dim, key_dim)         # query projection
        self.key_proj = nn.Linear(hidden_dim, key_dim)              # key projection
        self.value_proj = nn.Linear(hidden_dim, value_dim)          # value projection
        self.gate_proj = nn.Linear(hidden_dim, key_dim)             # gate projection
        self.out_proj = nn.Linear(value_dim, output_dim)                  # output projection

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: Tensor of shape (B, T, hidden_dim)

        Returns:
            output: Tensor of shape (B, T, output_dim)
            final_state: Tensor of shape (B, key_dim, value_dim)
        """
        B, T, _ = hidden_state.shape
        K, V, O = self.key_dim, self.value_dim, self.output_dim
        dtype = hidden_state.dtype
        device = hidden_state.device

        query = self.query_proj(hidden_state)                # (B, T, hidden_dim)
        key = F.sigmoid(self.key_proj(hidden_state))          # (B, T, key_dim)
        gate = F.sigmoid(self.gate_proj(hidden_state))        # (B, T, key_dim)
        value = self.value_proj(hidden_state)                 # (B, T, value_dim)

        state = torch.zeros(B, K, V, dtype=dtype, device=device)
        output = torch.zeros(B, T, O, dtype=dtype, device=device)

        for i in range(T):
            query_i = query[:, i]                    # (B, hidden_dim)
            key_i = key[:, i]                        # (B, key_dim)
            value_i = value[:, i]                    # (B, value_dim)
            gate_i = gate[:, i]                      # (B, key_dim)            

            key_value_i = key_i.unsqueeze(-1) * value_i.unsqueeze(1)  # (B, key_dim, value_dim)
            state = state * gate_i.unsqueeze(-1) + key_value_i      # (B, key_dim, value_dim)
            output[:, i] = self.out_proj((query_i.unsqueeze(-1) * state).sum(-2))   # (B, output_dim)
        
        return output, state

class LM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, key_dim: int, value_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Embedding layer to convert input_ids to hidden states
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Stack of SimpleRNN layers
        self.layers = nn.ModuleList([SimpleRNN(hidden_dim, key_dim, value_dim, output_dim)] + [SimpleRNN(output_dim, key_dim, value_dim, output_dim) for _ in range(num_layers-1)])

        # Final output projection after reduction
        self.lm_head = nn.Linear(output_dim, vocab_size)  # Output logits for the vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (B, T) containing token indices

        Returns:
            output: Tensor of shape (B, T, vocab_size) containing token logits
        """
        B, T = input_ids.shape

        hidden_state = self.embedding(input_ids)  # (B, T, hidden_dim)

        for layer in self.layers:
            hidden_state, _ = layer(hidden_state)  # Pass through each SimpleRNN layer

        output = self.lm_head(hidden_state)  # (B, T, vocab_size)

        return output


if __name__ == "__main__":

    num_layers = 2
    vocab_size = 16  
    
    # Use different tensor dim sizes to double-check there is no missmatch
    bs = 2
    seq_len = 4
    hidden_dim = 8
    key_dim = 16
    value_dim = 32
    output_dim = 64

    model = LM(vocab_size, hidden_dim, key_dim, value_dim, output_dim, num_layers)
    x = torch.randint(0, vocab_size, (bs, seq_len))
    model(x)
