#!/usr/bin/env python3
"""Command-line text generation script."""

import argparse
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.transformer_block import Transformer  
from experiments.eval import load_checkpoint, create_model_from_config


def greedy_generate(model, prompt, char2idx, idx2char, max_new_tokens=50, device='cpu'):
    """Generate text using greedy decoding (argmax) for maximum accuracy."""
    model.eval()
    
    # Convert prompt to indices
    context = [char2idx[c] for c in prompt if c in char2idx]
    if not context:
        return prompt + " [UNKNOWN CHARACTERS]"
    
    context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    generated = prompt
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits
            logits, _ = model(context, is_causal=True)
            
            # Use greedy decoding (argmax) - most likely next token
            next_token_logits = logits[0, -1, :]  # Last position, first batch
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            
            # Convert to character
            next_char = idx2char[next_token]
            generated += next_char
            
            # Stop at period if we complete a phrase
            if next_char == '.' and len(generated) > len(prompt) + 5:
                break
            
            # Update context
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            context = torch.cat([context, next_token_tensor], dim=1)
            
            # Keep context window manageable
            if context.size(1) > 64:
                context = context[:, -64:]
    
    return generated


def main():
    parser = argparse.ArgumentParser(description='Generate text from Transformer model')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt to complete')
    parser.add_argument('--ckpt', type=str, default='checkpoints/checkpoint_5000.pt', help='Checkpoint path')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum new tokens to generate')
    
    args = parser.parse_args()
    
    device = torch.device('cpu')
    
    # Load model
    try:
        checkpoint = load_checkpoint(args.ckpt, device)
        model = create_model_from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Get vocab
        char2idx = checkpoint['config']['vocab_info']['char2idx']
        idx2char = {int(k): v for k, v in checkpoint['config']['vocab_info']['idx2char'].items()}
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Generate text
    result = greedy_generate(
        model, args.prompt, char2idx, idx2char,
        max_new_tokens=args.max_new_tokens, device=device
    )
    
    # Display result
    print(f"Input:  '{args.prompt}'")
    print(f"Output: '{result}'")


if __name__ == "__main__":
    main()