"""Mini Transformer implementation."""

from .attention_numpy import scaled_dot_product_attention as np_attention
from .attention_torch import (
    scaled_dot_product_attention,
    ScaledDotProductAttention,
    create_padding_mask,
    create_causal_mask,
)
from .mha import MultiHeadAttention, SelfAttention, CrossAttention
from .positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    get_positional_encoding,
)
from .transformer_block import (
    FeedForward,
    TransformerBlock,
    Transformer,
)

__all__ = [
    # NumPy attention
    "np_attention",
    # PyTorch attention
    "scaled_dot_product_attention",
    "ScaledDotProductAttention",
    "create_padding_mask",
    "create_causal_mask",
    # Multi-head attention
    "MultiHeadAttention",
    "SelfAttention", 
    "CrossAttention",
    # Positional encoding
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "RotaryPositionalEncoding",
    "get_positional_encoding",
    # Transformer
    "FeedForward",
    "TransformerBlock",
    "Transformer",
]
