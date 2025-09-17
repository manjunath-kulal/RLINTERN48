"""Transformer block implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .mha import MultiHeadAttention, SelfAttention
from .positional_encoding import get_positional_encoding


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_p: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
    ):
        """
        Initialize Feed-Forward Network.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension in FFN
            dropout_p: Dropout probability
            activation: Activation function ("gelu", "relu", "swish")
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_p)
        
        # Set activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "swish" or activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of FFN."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        bias: bool = True,
    ):
        """
        Initialize Transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Hidden dimension in FFN (default: 4 * d_model)
            dropout_p: General dropout probability
            attention_dropout_p: Attention-specific dropout probability
            activation: Activation function in FFN
            norm_first: Whether to apply LayerNorm before (True) or after (False) sublayers
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_first = norm_first
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Self-attention
        self.self_attention = SelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout_p=attention_dropout_p,
            bias=bias,
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout_p=dropout_p,
            activation=activation,
            bias=bias,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask: Optional attention mask
            is_causal: Whether to apply causal masking
            return_attention: Whether to return attention weights
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Optional attention weights
        """
        if self.norm_first:
            # Pre-LayerNorm architecture
            attn_output, attn_weights = self._pre_norm_attention(
                x, attn_mask, is_causal, return_attention
            )
            output = self._pre_norm_ffn(attn_output)
        else:
            # Post-LayerNorm architecture  
            attn_output, attn_weights = self._post_norm_attention(
                x, attn_mask, is_causal, return_attention
            )
            output = self._post_norm_ffn(attn_output)
        
        return output, attn_weights
    
    def _pre_norm_attention(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        return_attention: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pre-norm self-attention sublayer."""
        # Normalize input
        normalized_x = self.norm1(x)
        
        # Self-attention
        attn_output, attn_weights = self.self_attention(
            normalized_x, attn_mask, is_causal, return_attention
        )
        
        # Residual connection with dropout
        attn_output = self.dropout1(attn_output)
        output = x + attn_output
        
        return output, attn_weights
    
    def _pre_norm_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm feed-forward sublayer."""
        # Normalize input
        normalized_x = self.norm2(x)
        
        # Feed-forward
        ffn_output = self.feed_forward(normalized_x)
        
        # Residual connection with dropout
        ffn_output = self.dropout2(ffn_output)
        output = x + ffn_output
        
        return output
    
    def _post_norm_attention(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        return_attention: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Post-norm self-attention sublayer."""
        # Self-attention
        attn_output, attn_weights = self.self_attention(
            x, attn_mask, is_causal, return_attention
        )
        
        # Residual connection with dropout and normalization
        attn_output = self.dropout1(attn_output)
        output = self.norm1(x + attn_output)
        
        return output, attn_weights
    
    def _post_norm_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Post-norm feed-forward sublayer."""
        # Feed-forward
        ffn_output = self.feed_forward(x)
        
        # Residual connection with dropout and normalization
        ffn_output = self.dropout2(ffn_output)
        output = self.norm2(x + ffn_output)
        
        return output


class Transformer(nn.Module):
    """Complete Transformer model for character-level language modeling."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: Optional[int] = None,
        max_len: int = 5000,
        dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        pos_encoding_type: str = "sinusoidal",
        activation: str = "gelu",
        norm_first: bool = True,
        tie_weights: bool = False,
        bias: bool = True,
    ):
        """
        Initialize Transformer model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            d_ff: Hidden dimension in FFN (default: 4 * d_model)
            max_len: Maximum sequence length
            dropout_p: General dropout probability
            attention_dropout_p: Attention-specific dropout probability
            pos_encoding_type: Type of positional encoding
            activation: Activation function in FFN
            norm_first: Whether to use pre-norm architecture
            tie_weights: Whether to tie input/output embeddings
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = get_positional_encoding(
            pos_encoding_type, d_model, max_len, dropout_p
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
                attention_dropout_p=attention_dropout_p,
                activation=activation,
                norm_first=norm_first,
                bias=bias,
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm (for pre-norm architecture)
        if norm_first:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = None
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie input/output embeddings if requested
        if tie_weights:
            self.output_projection.weight = self.token_embedding.weight
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # Output projection (if not tied)
        if not self.tie_weights:
            nn.init.normal_(self.output_projection.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Transformer.
        
        Args:
            input_ids: Input token ids of shape (batch_size, seq_len)
            attn_mask: Optional attention mask
            is_causal: Whether to apply causal masking
            return_attention: Whether to return attention weights
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            attention_weights: Optional attention weights from last layer
        """
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = x * (self.d_model ** 0.5)  # Scale embeddings
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer blocks
        attention_weights = None
        for i, block in enumerate(self.blocks):
            is_last = i == len(self.blocks) - 1
            x, attn_weights = block(
                x, attn_mask, is_causal, 
                return_attention=(return_attention and is_last)
            )
            if is_last:
                attention_weights = attn_weights
        
        # Final layer norm
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits, attention_weights


if __name__ == "__main__":
    # Test Transformer components
    torch.manual_seed(42)
    
    batch_size, seq_len, d_model = 2, 8, 64
    vocab_size, n_heads, n_layers = 100, 8, 2
    
    # Test FeedForward
    ffn = FeedForward(d_model, d_ff=256, dropout_p=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    ffn_out = ffn(x)
    print(f"FFN input: {x.shape}, output: {ffn_out.shape}")
    
    # Test TransformerBlock
    block = TransformerBlock(d_model, n_heads, dropout_p=0.1)
    block_out, attn_weights = block(x, is_causal=True, return_attention=True)
    print(f"Block input: {x.shape}, output: {block_out.shape}")
    print(f"Attention weights: {attn_weights.shape}")
    
    # Test complete Transformer
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout_p=0.1,
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, attn_weights = model(input_ids, return_attention=True)
    
    print(f"Model input: {input_ids.shape}, output: {logits.shape}")
    print(f"Final attention weights: {attn_weights.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
