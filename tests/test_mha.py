"""Tests for multi-head attention module."""

import torch
import unittest
from src.mha import MultiHeadAttention, SelfAttention, CrossAttention


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 64
        self.n_heads = 4
        
        # Create MHA module
        self.mha = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_p=0.1
        )
        
        # Create test tensors
        self.q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.k = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.v = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
    def test_mha_shapes(self):
        """Test that MHA produces correct output shapes."""
        output, attn_weights = self.mha(self.q, self.k, self.v)
        
        # Check output shape
        expected_output_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_output_shape)
        
        # Attention weights should be None by default
        self.assertIsNone(attn_weights)
        
    def test_mha_with_attention_return(self):
        """Test MHA with attention weights returned."""
        output, attn_weights = self.mha(self.q, self.k, self.v, return_attention=True)
        
        # Check output shape
        expected_output_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_output_shape)
        
        # Check attention weights shape
        expected_attn_shape = (self.batch_size, self.n_heads, self.seq_len, self.seq_len)
        self.assertEqual(attn_weights.shape, expected_attn_shape)
    
    def test_mha_gradients(self):
        """Test that gradients flow through MHA."""
        self.q.requires_grad_(True)
        self.k.requires_grad_(True)
        self.v.requires_grad_(True)
        
        output, _ = self.mha(self.q, self.k, self.v)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(self.q.grad)
        self.assertIsNotNone(self.k.grad)
        self.assertIsNotNone(self.v.grad)
        
        # Check gradient shapes
        self.assertEqual(self.q.grad.shape, self.q.shape)
        self.assertEqual(self.k.grad.shape, self.k.shape)
        self.assertEqual(self.v.grad.shape, self.v.shape)


if __name__ == "__main__":
    unittest.main()
