"""Scaled Dot-Product Attention implementation in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len_q, d_k)
        key: Key tensor of shape (batch_size, seq_len_k, d_k)
        value: Value tensor of shape (batch_size, seq_len_k, d_v)
        attn_mask: Optional attention mask of shape (batch_size, seq_len_q, seq_len_k)
        dropout_p: Dropout probability (for training)
        is_causal: Whether to apply causal (lower triangular) mask
        training: Whether in training mode (affects dropout)
    
    Returns:
        output: Attention output of shape (batch_size, seq_len_q, d_v)
        attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
    """
    batch_size, seq_len_q, d_k = query.shape
    seq_len_k = key.shape[1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Apply causal mask if requested
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=query.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply custom mask if provided
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))
    
    # Compute softmax attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout during training
    if training and dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p, training=training)
    
    # Compute output: attention_weights @ V
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


class ScaledDotProductAttention(nn.Module):
    """PyTorch module for Scaled Dot-Product Attention."""
    
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of attention mechanism."""
        return scaled_dot_product_attention(
            query, key, value, attn_mask, self.dropout_p, is_causal, self.training
        )


def create_padding_mask(
    sequences: torch.Tensor, pad_token_id: int = 0
) -> torch.Tensor:
    """
    Create padding mask for attention.
    
    Args:
        sequences: Input sequences of shape (batch_size, seq_len)
        pad_token_id: ID of padding token
    
    Returns:
        mask: Padding mask of shape (batch_size, 1, seq_len)
    """
    mask = (sequences != pad_token_id).float()
    return mask.unsqueeze(1)  # Add dimension for broadcasting


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal (lower triangular) mask for autoregressive attention.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
    
    Returns:
        mask: Causal mask of shape (seq_len, seq_len)
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


def create_look_ahead_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Create look-ahead mask to prevent attention to future positions."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0  # Return False for future positions


if __name__ == "__main__":
    # Simple test
    torch.manual_seed(42)
    
    batch_size, seq_len, d_model = 2, 4, 8
    device = torch.device("cpu")
    
    # Create random Q, K, V
    query = torch.randn(batch_size, seq_len, d_model, device=device)
    key = torch.randn(batch_size, seq_len, d_model, device=device)
    value = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Test attention
    output, weights = scaled_dot_product_attention(query, key, value, is_causal=True)
    
    print(f"Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights sum (should be ~1.0): {weights.sum(dim=-1)}")
    
    # Test module
    attention_module = ScaledDotProductAttention(dropout_p=0.1)
    output2, weights2 = attention_module(query, key, value, is_causal=True)
    print(f"Module output shape: {output2.shape}")
