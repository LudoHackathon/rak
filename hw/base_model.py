import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC
from typing import Tuple, Type


class BaseRNN(nn.Module, ABC):
    """
    Abstract class to make LM more readable and easy to test with different backbones
    """
    def __init__(self, hidden_dim: int, key_dim: int, value_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: Tensor of shape (B, T, hidden_dim)

        Returns:
            output: Tensor of shape (B, T, output_dim)
            final_state: Tensor of shape (B, key_dim, value_dim)
        """
        B, T, _ = hidden_state.shape
        dtype = hidden_state.dtype
        device = hidden_state.device

        query, key, value, gate = self._qkvg_proj(hidden_state)

        output, state = self._rnn_loop(query, key, value, gate, B, T, dtype, device)
        
        return output, state
    
    def _qkvg_proj(self, hidden_state):
        raise NotImplementedError
    
    def _rnn_loop(self, query, key, value, gate, B, T, dtype, device):
        raise NotImplementedError


class LM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, key_dim: int, value_dim: int, output_dim: int, num_layers: int, backbone: Type[BaseRNN]):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Embedding layer to convert input_ids to hidden states
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Stack of Backbone layers
        self.layers = nn.ModuleList([backbone(hidden_dim, key_dim, value_dim, output_dim)] + [backbone(output_dim, key_dim, value_dim, output_dim) for _ in range(num_layers)])

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
