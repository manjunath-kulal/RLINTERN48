"""Tests for attention implementations."""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from src.attention_numpy import (
    scaled_dot_product_attention as np_attention,
    softmax,
    create_padding_mask as np_create_padding_mask,
    create_causal_mask as np_create_causal_mask,
)
from src.attention_torch import (
    scaled_dot_product_attention,
    ScaledDotProductAttention,
    create_padding_mask,
    create_causal_mask,
)


class TestNumPyAttention:
    """Test NumPy attention implementation."""
    
    def test_attention_shapes(self):
        """Test attention output shapes."""
        np.random.seed(42)
        batch_size, seq_len, d_model = 2, 4, 8
        
        Q = np.random.randn(batch_size, seq_len, d_model)
        K = np.random.randn(batch_size, seq_len, d_model)
        V = np.random.randn(batch_size, seq_len, d_model)
        
        output, weights = np_attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        np.random.seed(42)
        batch_size, seq_len, d_model = 2, 4, 8
        
        Q = np.random.randn(batch_size, seq_len, d_model)
        K = np.random.randn(batch_size, seq_len, d_model)
        V = np.random.randn(batch_size, seq_len, d_model)
        
        _, weights = np_attention(Q, K, V)
        
        # Weights should sum to 1 along last dimension
        weight_sums = np.sum(weights, axis=-1)
        np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-6)
    
    def test_causal_masking(self):
        """Test causal masking."""
        np.random.seed(42)
        batch_size, seq_len, d_model = 1, 4, 8
        
        Q = np.random.randn(batch_size, seq_len, d_model)
        K = np.random.randn(batch_size, seq_len, d_model)
        V = np.random.randn(batch_size, seq_len, d_model)
        
        _, weights = np_attention(Q, K, V, is_causal=True)
        
        # Upper triangular part should be 0 (after softmax of -inf)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[0, i, j] == 0.0
    
    def test_softmax(self):
        """Test softmax implementation."""
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = softmax(x, axis=-1)
        
        # Should sum to 1
        np.testing.assert_allclose(result.sum(axis=-1), 1.0)
        
        # Should match expected values
        expected = np.array([
            [0.09003057, 0.24472847, 0.66524096],
            [0.33333333, 0.33333333, 0.33333333]
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_padding_mask(self):
        """Test padding mask creation."""
        sequences = np.array([[1, 2, 3, 0], [1, 2, 0, 0]])
        mask = np_create_padding_mask(sequences, pad_token_id=0)
        
        expected = np.array([[[1, 1, 1, 0]], [[1, 1, 0, 0]]])
        np.testing.assert_array_equal(mask, expected)
    
    def test_causal_mask_creation(self):
        """Test causal mask creation."""
        mask = np_create_causal_mask(3)
        expected = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ])
        np.testing.assert_array_equal(mask, expected)


class TestPyTorchAttention:
    """Test PyTorch attention implementation."""
    
    def test_attention_shapes(self):
        """Test attention output shapes."""
        torch.manual_seed(42)
        batch_size, seq_len, d_model = 2, 4, 8
        
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)
        
        output, weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        torch.manual_seed(42)
        batch_size, seq_len, d_model = 2, 4, 8
        
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)
        
        _, weights = scaled_dot_product_attention(Q, K, V)
        
        # Weights should sum to 1 along last dimension
        weight_sums = weights.sum(dim=-1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums))
    
    def test_causal_masking(self):
        """Test causal masking."""
        torch.manual_seed(42)
        batch_size, seq_len, d_model = 1, 4, 8
        
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)
        
        _, weights = scaled_dot_product_attention(Q, K, V, is_causal=True)
        
        # Upper triangular part should be 0 (after softmax of -inf)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[0, i, j].item() == 0.0
    
    def test_attention_module(self):
        """Test attention module."""
        torch.manual_seed(42)
        batch_size, seq_len, d_model = 2, 4, 8
        
        attention = ScaledDotProductAttention(dropout_p=0.1)
        
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)
        
        output, weights = attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_gradients(self):
        """Test that gradients flow through attention."""
        torch.manual_seed(42)
        batch_size, seq_len, d_model = 1, 3, 4
        
        Q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        output, _ = scaled_dot_product_attention(Q, K, V)
        loss = output.sum()
        loss.backward()
        
        # Gradients should exist and not be zero
        assert Q.grad is not None
        assert K.grad is not None  
        assert V.grad is not None
        assert not torch.allclose(Q.grad, torch.zeros_like(Q.grad))
        assert not torch.allclose(K.grad, torch.zeros_like(K.grad))
        assert not torch.allclose(V.grad, torch.zeros_like(V.grad))
    
    def test_padding_mask(self):
        """Test padding mask creation."""
        sequences = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]])
        mask = create_padding_mask(sequences, pad_token_id=0)
        
        expected = torch.tensor([[[1, 1, 1, 0]], [[1, 1, 0, 0]]], dtype=torch.float32)
        torch.testing.assert_close(mask, expected)
    
    def test_causal_mask_creation(self):
        """Test causal mask creation."""
        mask = create_causal_mask(3)
        expected = torch.tensor([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ], dtype=torch.float32)
        torch.testing.assert_close(mask, expected)


class TestAttentionConsistency:
    """Test consistency between NumPy and PyTorch implementations."""
    
    def test_numpy_pytorch_consistency(self):
        """Test that NumPy and PyTorch give similar results."""
        # Set seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        batch_size, seq_len, d_model = 1, 3, 4
        
        # Create same inputs for both implementations
        Q_np = np.random.randn(batch_size, seq_len, d_model)
        K_np = np.random.randn(batch_size, seq_len, d_model)
        V_np = np.random.randn(batch_size, seq_len, d_model)
        
        Q_torch = torch.from_numpy(Q_np).float()
        K_torch = torch.from_numpy(K_np).float()
        V_torch = torch.from_numpy(V_np).float()
        
        # Compute attention
        output_np, weights_np = np_attention(Q_np, K_np, V_np, training=False)
        output_torch, weights_torch = scaled_dot_product_attention(
            Q_torch, K_torch, V_torch, training=False
        )
        
        # Convert PyTorch results to NumPy
        output_torch_np = output_torch.detach().numpy()
        weights_torch_np = weights_torch.detach().numpy()
        
        # Check if results are close
        np.testing.assert_allclose(output_np, output_torch_np, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights_np, weights_torch_np, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
