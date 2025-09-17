#!/usr/bin/env python3
"""Quick greedy generation script for maximum accuracy."""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.transformer_block import Transformer
from experiments.eval import load_checkpoint, create_model_from_config

def greedy_generate(model, prompt, char2idx, idx2char, max_new_tokens=50, device='cpu'):
    """Generate text using greedy decoding (argmax) for maximum accuracy."""
    model.eval()
    
    # Convert prompt to indices
    context = [char2idx[c] for c in prompt if c in char2idx]
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
            if next_char == '.' and len(generated) > len(prompt) + 10:
                break
            
            # Update context
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            context = torch.cat([context, next_token_tensor], dim=1)
            
            # Keep context window manageable
            if context.size(1) > 64:
                context = context[:, -64:]
    
    return generated

def main():
    device = torch.device('cpu')
    
    # Load model
    checkpoint = load_checkpoint('checkpoints_finetune/last.pt', device)
    model = create_model_from_config(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Get vocab
    char2idx = checkpoint['config']['vocab_info']['char2idx']
    idx2char = {int(k): v for k, v in checkpoint['config']['vocab_info']['idx2char'].items()}
    
    # Test examples
    test_prompts = ["Better", "bird", "The early", "Practice", "All"]
    
    print("🎯 GREEDY GENERATION (Maximum Accuracy)")
    print("=" * 50)
    
    for prompt in test_prompts:
        result = greedy_generate(model, prompt, char2idx, idx2char, max_new_tokens=30, device=device)
        print(f"Input: '{prompt}'")
        print(f"Output: '{result}'")
        print("-" * 50)

if __name__ == "__main__":
    main()