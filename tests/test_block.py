"""Tests for transformer block module."""

import torch
import unittest
from src.transformer_block import FeedForward, TransformerBlock, Transformer


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 64
        self.d_ff = 256
        
        self.ff = FeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout_p=0.1)
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
    def test_ff_shapes(self):
        """Test FeedForward produces correct shapes."""
        output = self.ff(self.x)
        
        # Should preserve input shape
        self.assertEqual(output.shape, self.x.shape)
        
    def test_ff_gradients(self):
        """Test FeedForward gradients flow properly."""
        self.x.requires_grad_(True)
        
        output = self.ff(self.x)
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients
        self.assertIsNotNone(self.x.grad)
        self.assertEqual(self.x.grad.shape, self.x.shape)


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 64
        self.n_heads = 4
        self.d_ff = 256
        
        self.block = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout_p=0.1,
            norm_first=True
        )
        
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
    def test_transformer_block_shapes(self):
        """Test TransformerBlock produces correct shapes."""
        output, attn_weights = self.block(self.x)
        
        # Should preserve input shape
        self.assertEqual(output.shape, self.x.shape)
        
        # Attention weights should be None by default
        self.assertIsNone(attn_weights)
        
    def test_transformer_block_with_causal(self):
        """Test TransformerBlock with causal attention."""
        output, attn_weights = self.block(self.x, is_causal=True)
        
        # Should preserve input shape
        self.assertEqual(output.shape, self.x.shape)
        
    def test_transformer_block_gradients(self):
        """Test TransformerBlock gradients flow properly."""
        self.x.requires_grad_(True)
        
        output, _ = self.block(self.x)
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients
        self.assertIsNotNone(self.x.grad)
        self.assertEqual(self.x.grad.shape, self.x.shape)
        
    def test_pre_norm_vs_post_norm(self):
        """Test pre-norm vs post-norm configurations."""
        # Pre-norm block
        block_pre = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout_p=0.0,  # No dropout for deterministic comparison
            norm_first=True
        )
        
        # Post-norm block
        block_post = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout_p=0.0,  # No dropout for deterministic comparison
            norm_first=False
        )
        
        # Both should produce same shape but different values
        output_pre, _ = block_pre(self.x)
        output_post, _ = block_post(self.x)
        
        self.assertEqual(output_pre.shape, output_post.shape)
        # Should be different due to different normalization order
        self.assertFalse(torch.allclose(output_pre, output_post))


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        self.vocab_size = 100
        self.d_model = 64
        self.n_heads = 4
        self.n_layers = 2
        self.d_ff = 256
        
        self.transformer = Transformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            max_len=512,
            dropout_p=0.1,
            norm_first=True,
            pos_encoding_type='sinusoidal'
        )
        
        # Create token indices
        self.tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
    def test_transformer_shapes(self):
        """Test Transformer produces correct shapes."""
        output, all_attn_weights = self.transformer(self.tokens)
        
        # Should produce logits for each token position
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(output.shape, expected_shape)
        
    def test_transformer_causal_mask(self):
        """Test Transformer with causal mask."""
        output, _ = self.transformer(self.tokens, is_causal=True)
        
        # Should preserve output shape
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(output.shape, expected_shape)
        
    def test_transformer_gradients(self):
        """Test Transformer gradients flow properly."""
        output, _ = self.transformer(self.tokens)
        loss = output.sum()
        loss.backward()
        
        # Check that embedding gradients exist
        self.assertIsNotNone(self.transformer.token_embedding.weight.grad)
        
    def test_transformer_parameter_count(self):
        """Test Transformer has reasonable parameter count."""
        total_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        
        # Should have non-zero parameters
        self.assertGreater(total_params, 0)
        
        # Print for debugging
        print(f"Total parameters: {total_params:,}")
        
    def test_different_pos_encodings(self):
        """Test different positional encoding types."""
        pos_encodings = ['sinusoidal', 'learned', 'rope']
        
        for pos_enc in pos_encodings:
            transformer = Transformer(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=1,  # Smaller for faster test
                d_ff=self.d_ff,
                max_len=512,
                dropout_p=0.0,
                norm_first=True,
                pos_encoding_type=pos_enc
            )
            
            output, _ = transformer(self.tokens)
            expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
            self.assertEqual(output.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
