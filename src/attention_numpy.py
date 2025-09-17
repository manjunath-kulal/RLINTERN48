"""Scaled Dot-Product Attention implementation in NumPy."""

import numpy as np
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    training: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Scaled Dot-Product Attention.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len_q, d_k)
        key: Key tensor of shape (batch_size, seq_len_k, d_k)
        value: Value tensor of shape (batch_size, seq_len_k, d_v)
        mask: Optional attention mask of shape (batch_size, seq_len_q, seq_len_k)
        dropout_p: Dropout probability (for training)
        is_causal: Whether to apply causal (lower triangular) mask
        training: Whether in training mode (affects dropout)
    
    Returns:
        output: Attention output of shape (batch_size, seq_len_q, d_v)
        attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
    """
    batch_size, seq_len_q, d_k = query.shape
    seq_len_k = key.shape[1]
    d_v = value.shape[2]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply causal mask if requested
    if is_causal:
        causal_mask = np.triu(np.ones((seq_len_q, seq_len_k)), k=1).astype(bool)
        scores = np.where(causal_mask[None, :, :], -np.inf, scores)
    
    # Apply custom mask if provided
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    
    # Compute softmax attention weights
    attention_weights = softmax(scores, axis=-1)
    
    # Apply dropout during training
    if training and dropout_p > 0.0:
        attention_weights = dropout(attention_weights, dropout_p)
    
    # Compute output: attention_weights @ V
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def dropout(x: np.ndarray, p: float) -> np.ndarray:
    """Apply dropout to input tensor."""
    if p == 0.0:
        return x
    keep_prob = 1.0 - p
    mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
    return x * mask


def create_padding_mask(
    sequences: np.ndarray, pad_token_id: int = 0
) -> np.ndarray:
    """
    Create padding mask for attention.
    
    Args:
        sequences: Input sequences of shape (batch_size, seq_len)
        pad_token_id: ID of padding token
    
    Returns:
        mask: Padding mask of shape (batch_size, 1, seq_len)
    """
    mask = (sequences != pad_token_id).astype(np.float32)
    return mask[:, None, :]  # Add dimension for broadcasting


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal (lower triangular) mask for autoregressive attention.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        mask: Causal mask of shape (seq_len, seq_len)
    """
    return np.tril(np.ones((seq_len, seq_len))).astype(np.float32)


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    
    batch_size, seq_len, d_model = 2, 4, 8
    
    # Create random Q, K, V
    query = np.random.randn(batch_size, seq_len, d_model)
    key = np.random.randn(batch_size, seq_len, d_model)
    value = np.random.randn(batch_size, seq_len, d_model)
    
    # Test attention
    output, weights = scaled_dot_product_attention(query, key, value, is_causal=True)
    
    print(f"Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights sum (should be ~1.0): {weights.sum(axis=-1)}")
