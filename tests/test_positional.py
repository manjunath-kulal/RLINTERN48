"""Tests for positional encoding modules."""

import torch
import unittest
import math
from src.positional_encoding import (
    SinusoidalPositionalEncoding, 
    LearnedPositionalEncoding,
    RotaryPositionalEncoding
)


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 64
        self.pe = SinusoidalPositionalEncoding(d_model=self.d_model, max_len=100)
        
    def test_sinusoidal_pe_shapes(self):
        """Test sinusoidal PE produces correct shapes."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.pe(x)
        
        # Should preserve input shape
        self.assertEqual(output.shape, x.shape)
        
    def test_sinusoidal_pe_deterministic(self):
        """Test that sinusoidal PE is deterministic."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output1 = self.pe(x)
        output2 = self.pe(x)
        
        # Should be identical
        self.assertTrue(torch.allclose(output1, output2))
        
    def test_sinusoidal_pe_values(self):
        """Test sinusoidal PE values are reasonable."""
        x = torch.zeros(1, self.seq_len, self.d_model)
        output = self.pe(x)
        
        # PE should be added to input, so output != input for zero input
        self.assertFalse(torch.allclose(output, x))
        
        # PE values should be bounded
        pe_values = output - x  # Extract just the PE
        self.assertTrue(torch.all(pe_values >= -1.1))
        self.assertTrue(torch.all(pe_values <= 1.1))


class TestLearnedPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 64
        self.pe = LearnedPositionalEncoding(d_model=self.d_model, max_len=100)
        
    def test_learned_pe_shapes(self):
        """Test learned PE produces correct shapes."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.pe(x)
        
        # Should preserve input shape
        self.assertEqual(output.shape, x.shape)
        
    def test_learned_pe_gradients(self):
        """Test learned PE parameters have gradients."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x.requires_grad_(True)
        
        output = self.pe(x)
        loss = output.sum()
        loss.backward()
        
        # PE parameters should have gradients
        self.assertIsNotNone(self.pe.position_embedding.weight.grad)
        
    def test_learned_pe_parameters(self):
        """Test learned PE has correct number of parameters."""
        # Should have max_len x d_model parameters
        expected_params = 100 * self.d_model
        actual_params = sum(p.numel() for p in self.pe.parameters())
        self.assertEqual(actual_params, expected_params)


class TestRotaryPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 64
        self.n_heads = 4
        self.pe = RotaryPositionalEncoding(d_model=self.d_model)
        
    def test_rope_shapes(self):
        """Test RoPE produces correct shapes."""
        # RoPE expects input in standard format (batch, seq_len, d_model)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.pe(x)
        
        # Should preserve input shape
        self.assertEqual(output.shape, x.shape)
        
    def test_rope_rotation_property(self):
        """Test RoPE rotation property."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Apply RoPE
        output = self.pe(x)
        
        # Should be different from input (rotated)
        self.assertFalse(torch.allclose(output, x))
        
        # Magnitude should be preserved approximately (rotation property)
        input_norm = torch.norm(x, dim=-1)
        output_norm = torch.norm(output, dim=-1)
        self.assertTrue(torch.allclose(input_norm, output_norm, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
