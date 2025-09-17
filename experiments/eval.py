#!/usr/bin/env python3
"""Model evaluation script."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from src.transformer_block import Transformer
from experiments.train_charlm import CharDataset, load_data, evaluate, bits_per_character


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def create_model_from_config(config: Dict[str, Any]) -> Transformer:
    """Create model from configuration."""
    model_config = config['model_config']
    return Transformer(**model_config)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    vocab_info: Dict[str, Any],
) -> Dict[str, float]:
    """Comprehensive model evaluation."""
    model.eval()
    
    # Basic evaluation
    avg_loss, perplexity = evaluate(model, data_loader, device)
    bpc = bits_per_character(avg_loss)
    
    # Token-level accuracy
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            
            logits, _ = model(x, is_causal=True)
            predictions = torch.argmax(logits, dim=-1)
            
            correct_predictions += (predictions == y).sum().item()
            total_predictions += y.numel()
    
    accuracy = correct_predictions / total_predictions
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'bits_per_character': bpc,
        'accuracy': accuracy,
        'vocab_size': vocab_info['vocab_size'],
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--data', type=str, default=None, help='Evaluation data path (default: use training data)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = load_checkpoint(args.ckpt, device)
    config = checkpoint['config']
    
    # Get data path
    if args.data is None:
        args.data = config['training_args']['data']
    
    print(f"Loading data: {args.data}")
    
    # Load data using same sequence length as training
    seq_len = config['training_args']['seq_len']
    _, val_loader, vocab_info = load_data(
        args.data, seq_len, args.batch_size, train_split=0.9
    )
    
    # Verify vocab consistency
    if vocab_info['vocab_size'] != config['vocab_info']['vocab_size']:
        print("Warning: Vocabulary size mismatch between training and evaluation data!")
        print(f"Training vocab size: {config['vocab_info']['vocab_size']}")
        print(f"Evaluation vocab size: {vocab_info['vocab_size']}")
    
    # Create and load model
    print("Creating model...")
    model = create_model_from_config(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, val_loader, device, vocab_info)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Bits per Character: {results['bits_per_character']:.4f}")
    print(f"Token Accuracy: {results['accuracy']:.4f}")
    print(f"Vocabulary Size: {results['vocab_size']}")
    
    # Check target performance
    target_bpc = 1.35
    if results['bits_per_character'] <= target_bpc:
        print(f"\n✅ PASS: BPC ({results['bits_per_character']:.4f}) ≤ target ({target_bpc})")
    else:
        print(f"\n❌ FAIL: BPC ({results['bits_per_character']:.4f}) > target ({target_bpc})")
    
    print("="*50)
    
    # Save results
    results_path = os.path.join(os.path.dirname(args.ckpt), 'eval_results.json')
    results['checkpoint_path'] = args.ckpt
    results['training_step'] = checkpoint['step']
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
