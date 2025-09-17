"""Multi-Head Attention implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention_torch import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_p: float = 0.0,
        bias: bool = True,
    ):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout_p: Dropout probability
            bias: Whether to use bias in linear projections
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_p = dropout_p
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout_p)
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_k, d_model)
            attn_mask: Optional attention mask
            is_causal: Whether to apply causal masking
            return_attention: Whether to return attention weights
        
        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Optional attention weights (if return_attention=True)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Linear projections
        Q = self.w_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.w_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.w_v(value)  # (batch_size, seq_len_k, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: (batch_size, n_heads, seq_len, d_k)
        
        # Apply attention to each head
        attn_output, attn_weights = self._multi_head_attention(
            Q, K, V, attn_mask, is_causal
        )
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Apply output projection
        output = self.w_o(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output, None
    
    def _multi_head_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention to multiple heads."""
        batch_size, n_heads, seq_len_q, d_k = Q.shape
        seq_len_k = K.shape[2]
        
        # Reshape for batch attention computation
        Q = Q.contiguous().view(batch_size * n_heads, seq_len_q, d_k)
        K = K.contiguous().view(batch_size * n_heads, seq_len_k, d_k)
        V = V.contiguous().view(batch_size * n_heads, seq_len_k, d_k)
        
        # Expand mask for multiple heads if provided
        if attn_mask is not None:
            if attn_mask.dim() == 3:  # (batch_size, seq_len_q, seq_len_k)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
            attn_mask = attn_mask.view(batch_size * n_heads, seq_len_q, seq_len_k)
        
        # Apply scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, attn_mask, self.dropout_p, is_causal, self.training
        )
        
        # Reshape back to multi-head format
        attn_output = attn_output.view(batch_size, n_heads, seq_len_q, d_k)
        attn_weights = attn_weights.view(batch_size, n_heads, seq_len_q, seq_len_k)
        
        return attn_output, attn_weights


class SelfAttention(MultiHeadAttention):
    """Self-Attention layer (Q, K, V from same input)."""
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of self-attention."""
        return super().forward(x, x, x, attn_mask, is_causal, return_attention)


class CrossAttention(MultiHeadAttention):
    """Cross-Attention layer (Q from one source, K, V from another)."""
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of cross-attention."""
        return super().forward(
            query, key_value, key_value, attn_mask, is_causal=False, return_attention=return_attention
        )


if __name__ == "__main__":
    # Test Multi-Head Attention
    torch.manual_seed(42)
    
    batch_size, seq_len, d_model, n_heads = 2, 8, 64, 8
    device = torch.device("cpu")
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Test Multi-Head Attention
    mha = MultiHeadAttention(d_model, n_heads, dropout_p=0.1)
    output, attn_weights = mha(x, x, x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test Self-Attention
    self_attn = SelfAttention(d_model, n_heads)
    output_self, _ = self_attn(x, is_causal=True)
    print(f"Self-attention output shape: {output_self.shape}")
    
    # Test with single head (should match regular attention)
    single_head_mha = MultiHeadAttention(d_model, 1)
    output_single, _ = single_head_mha(x, x, x)
    print(f"Single head output shape: {output_single.shape}")
