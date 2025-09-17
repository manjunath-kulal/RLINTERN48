#!/usr/bin/env python3
"""Character-level language model training script."""

import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer_block import Transformer


class CharDataset(Dataset):
    """Character-level dataset."""
    
    def __init__(self, text: str, seq_len: int):
        self.seq_len = seq_len
        
        # Create character vocabulary
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert text to indices
        self.data = [self.char2idx[ch] for ch in text]
        
        print(f"Dataset: {len(text)} characters, {self.vocab_size} unique")
        print(f"Vocabulary: {''.join(chars)}")
    
    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input and target are offset by 1 for language modeling
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_data(data_path: str, seq_len: int, batch_size: int, train_split: float = 0.9) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Load and prepare data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset
    dataset = CharDataset(text, seq_len)
    
    # Train/validation split
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    
    vocab_info = {
        'vocab_size': dataset.vocab_size,
        'char2idx': dataset.char2idx,
        'idx2char': dataset.idx2char,
    }
    
    return train_loader, val_loader, vocab_info


def create_model(vocab_size: int, **model_kwargs) -> Transformer:
    """Create Transformer model."""
    return Transformer(vocab_size=vocab_size, **model_kwargs)


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Single training step."""
    model.train()
    x, y = batch
    x, y = x.to(device), y.to(device)
    
    # Forward pass
    logits, _ = model(x, is_causal=True)
    
    # Compute loss
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            
            logits, _ = model(x, is_causal=True)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += y.numel()
    
    if total_tokens == 0:
        return float("inf"), float("inf")
    if total_tokens == 0:
        return float('inf'), float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def bits_per_character(avg_loss: float) -> float:
    """Convert cross-entropy loss to bits per character."""
    return avg_loss / torch.log(torch.tensor(2.0)).item()


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    loss: float,
    config: Dict[str, Any],
    save_path: str,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss,
        'config': config,
    }
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/tiny.txt', help='Training data path')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--d_model', type=int, default=192, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=3, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--d_ff', type=int, default=None, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save interval')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, vocab_info = load_data(
        args.data, args.seq_len, args.batch_size
    )
    
    # Model configuration
    model_config = {
        'vocab_size': vocab_info['vocab_size'],
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_ff,
        'max_len': args.seq_len * 2,  # Allow longer sequences
        'dropout_p': args.dropout,
        'pos_encoding_type': 'sinusoidal',
    }
    
    # Create model
    print("Creating model...")
    model = create_model(**model_config)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    val_bpcs = []
    
    step = 0
    best_val_loss = float('inf')
    
    # Save initial configuration
    config = {
        'model_config': model_config,
        'training_args': vars(args),
        'vocab_info': vocab_info,
    }
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    train_iter = iter(train_loader)
    
    for step in tqdm(range(args.steps), desc="Training"):
        # Get next batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Training step
        train_loss = train_step(model, batch, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluation
        if (step + 1) % args.eval_interval == 0:
            val_loss, val_perplexity = evaluate(model, val_loader, device)
            val_bpc = bits_per_character(val_loss)
            
            val_losses.append(val_loss)
            val_bpcs.append(val_bpc)
            
            print(f"\nStep {step + 1}:")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val loss: {val_loss:.4f}")
            print(f"  Val perplexity: {val_perplexity:.2f}")
            print(f"  Val BPC: {val_bpc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, step + 1, val_loss, config,
                    os.path.join(args.output_dir, 'best.pt')
                )
        
        # Save checkpoint
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, step + 1, train_loss, config,
                os.path.join(args.output_dir, f'checkpoint_{step + 1}.pt')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.steps, train_losses[-1], config,
        os.path.join(args.output_dir, 'last.pt')
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    eval_steps = range(args.eval_interval - 1, args.steps, args.eval_interval)
    plt.plot(eval_steps, val_bpcs)
    plt.title('Validation BPC')
    plt.xlabel('Step')
    plt.ylabel('Bits per Character')
    plt.axhline(y=1.35, color='r', linestyle='--', label='Target BPC ≤ 1.35')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    plt.show()
    
    print(f"\nTraining completed! Best validation BPC: {min(val_bpcs):.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
