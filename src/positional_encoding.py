"""Positional encoding implementations."""

import math
import torch
import torch.nn as nn
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout_p: float = 0.0,
    ):
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout_p: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_p)
        
        # Create positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for even and odd dimensions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not parameter)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding using embedding layer."""
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout_p: float = 0.0,
    ):
        """
        Initialize learned positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout_p: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout_p)
        
        # Learnable position embeddings
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initialize position embeddings."""
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Optional position indices of shape (batch_size, seq_len)
                      If None, uses sequential positions 0, 1, 2, ...
        
        Returns:
            Output tensor with positional encoding added
        """
        batch_size, seq_len, _ = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        pos_embed = self.position_embedding(positions)
        x = x + pos_embed
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Embedding (RoPE) implementation."""
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        base: float = 10000.0,
    ):
        """
        Initialize RoPE.
        
        Args:
            d_model: Model dimension (must be even)
            max_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Compute frequency for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute rotary embeddings
        self._build_cache(max_len)
    
    def _build_cache(self, max_len: int) -> None:
        """Build cache for rotary embeddings."""
        position = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(position, self.inv_freq)
        
        # Create complex representation for rotation
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            start_pos: Starting position for positional encoding
        
        Returns:
            Output tensor with RoPE applied
        """
        seq_len = x.shape[-2]
        end_pos = start_pos + seq_len
        
        # Get frequencies for current positions
        cos = self.freqs_cos[start_pos:end_pos]
        sin = self.freqs_sin[start_pos:end_pos]
        
        # Reshape x for rotation: (..., seq_len, d_model/2, 2)
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        
        # Apply rotation
        x1, x2 = x_reshaped.unbind(-1)
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Reshape back
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return rotated.view(*x.shape)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Forward pass applying RoPE."""
        return self.apply_rotary_pos_emb(x, start_pos)


def get_positional_encoding(
    encoding_type: str,
    d_model: int,
    max_len: int = 5000,
    dropout_p: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to get positional encoding.
    
    Args:
        encoding_type: Type of encoding ('sinusoidal', 'learned', 'rope')
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout_p: Dropout probability
        **kwargs: Additional arguments for specific encodings
    
    Returns:
        Positional encoding module
    """
    if encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len, dropout_p)
    elif encoding_type == "learned":
        return LearnedPositionalEncoding(d_model, max_len, dropout_p)
    elif encoding_type == "rope":
        base = kwargs.get("base", 10000.0)
        return RotaryPositionalEncoding(d_model, max_len, base)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


if __name__ == "__main__":
    # Test positional encodings
    torch.manual_seed(42)
    
    batch_size, seq_len, d_model = 2, 10, 64
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    
    # Test sinusoidal encoding
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model, dropout_p=0.1)
    x_sin = sinusoidal_pe(x)
    print(f"Sinusoidal PE output shape: {x_sin.shape}")
    
    # Test learned encoding
    learned_pe = LearnedPositionalEncoding(d_model, dropout_p=0.1)
    x_learned = learned_pe(x)
    print(f"Learned PE output shape: {x_learned.shape}")
    
    # Test RoPE
    rope = RotaryPositionalEncoding(d_model)
    x_rope = rope(x)
    print(f"RoPE output shape: {x_rope.shape}")
    
    # Test factory function
    pe = get_positional_encoding("sinusoidal", d_model)
    x_factory = pe(x)
    print(f"Factory PE output shape: {x_factory.shape}")
