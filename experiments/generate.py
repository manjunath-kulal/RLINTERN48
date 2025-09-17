#!/usr/bin/env python3
"""Text generation script."""

import argparse
import math
import pathlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))

from src.positional_encoding import sinusoidal_positional_encoding
from src.transformer_block import TransformerBlock


class CharTransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=512):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, d_model)
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x):
        b, l = x.size()
        pos_emb = sinusoidal_positional_encoding(
            l, self.token_emb.embedding_dim, x.device
        )
        h = self.token_emb(x) + pos_emb.unsqueeze(0)
        attns = []
        for blk in self.blocks:
            h, attn = blk(h, causal=True)
            attns.append(attn)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, attns


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}
    args = ckpt["args"]

    model = CharTransformerLM(
        vocab_size=len(vocab),
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        d_ff=4 * args["d_model"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, stoi, itos


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """
    logits: [1, vocab_size]
    returns: [1,1] next token index
    """
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)

    if top_k is not None:
        topk_vals, topk_idx = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs)
        probs[0, topk_idx[0]] = topk_vals[0]

    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[:, 0] = 0  # always keep top token
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs)
        probs.scatter_(1, sorted_idx, sorted_probs)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    
    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def generate_text(
    model: torch.nn.Module,
    prompt: str,
    char2idx: Dict[str, int],
    idx2char: Dict[int, str],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: torch.device = torch.device('cpu'),
) -> str:
    """Generate text from a prompt."""
    model.eval()
    
    # Convert prompt to token indices
    if prompt:
        context = [char2idx.get(c, 0) for c in prompt]
    else:
        # Start with a random character if no prompt
        context = [torch.randint(0, len(char2idx), (1,)).item()]
    
    context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = model(context, is_causal=True)
            
            # Get logits for the last position
            next_token_logits = logits[0, -1, :]
            
            # Sample next token
            next_token = sample_from_logits(
                next_token_logits, temperature, top_k, top_p
            )
            
            # Add to context and generated tokens
            context = torch.cat([
                context,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
            
            generated_tokens.append(next_token)
            
            # Optional: truncate context to prevent memory issues
            if context.size(1) > 512:  # Keep last 512 tokens
                context = context[:, -512:]
    
    # Convert tokens back to text
    generated_text = ''.join([idx2char.get(token, '?') for token in generated_tokens])
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained model')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--prompt', type=str, default='', help='Text prompt')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
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
    vocab_info = config['vocab_info']
    
    # Create and load model
    print("Creating model...")
    model = create_model_from_config(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Print generation settings
    print("\n" + "="*50)
    print("GENERATION SETTINGS")
    print("="*50)
    print(f"Prompt: '{args.prompt}'")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Number of samples: {args.num_samples}")
    print("="*50)
    
    # Generate samples
    char2idx = vocab_info['char2idx']
    idx2char = vocab_info['idx2char']
    
    for i in range(args.num_samples):
        print(f"\nSample {i + 1}:")
        print("-" * 30)
        
        generated_text = generate_text(
            model=model,
            prompt=args.prompt,
            char2idx=char2idx,
            idx2char=idx2char,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        
        # Print prompt + generated text
        full_text = args.prompt + generated_text
        print(full_text)
        print()


if __name__ == "__main__":
    main()
