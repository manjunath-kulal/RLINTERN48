#!/usr/bin/env python3
"""Extended testing with many more input prompts."""

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
    device = torch.device('cpu')
    
    # Load model
    checkpoint = load_checkpoint('checkpoints_finetune/last.pt', device)
    model = create_model_from_config(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Get vocab
    char2idx = checkpoint['config']['vocab_info']['char2idx']
    idx2char = {int(k): v for k, v in checkpoint['config']['vocab_info']['idx2char'].items()}
    
    # Extensive test prompts
    test_prompts = [
        # Original examples
        "Better",
        "bird", 
        "The early",
        
        # Complete phrase beginnings
        "Practice",
        "All that",
        "The pen",
        "Time heals",
        "Actions speak",
        "Beauty is",
        "Knowledge is",
        "Fortune favors",
        "When in Rome",
        "Don't count",
        "A picture",
        "Rome wasn't",
        "All roads",
        "The grass",
        "You can't judge",
        
        # Partial words
        "Prac",
        "Know",
        "Fort",
        "Beau",
        "glitt",
        "might",
        "Roma",
        "chick",
        
        # Single letters
        "T",
        "A",
        "B",
        "P",
        "K",
        
        # Mid-word starts
        "quick",
        "brown",
        "lazy",
        "journey",
        "thousand",
        "single",
        "question",
        "gold",
        "sword",
        "wounds",
        "louder",
        "beholder",
        "never",
        "perfect",
        "power",
        "bold",
        "Romans",
        "catches",
        "worm",
        "count",
        "chickens",
        "hatch",
        "picture",
        "worth",
        "words",
        "built",
        "roads",
        "lead",
        "grass",
        "greener",
        "side",
        "judge",
        "book",
        "cover"
    ]
    
    print("🎯 EXTENSIVE TESTING - GREEDY GENERATION")
    print("=" * 80)
    
    perfect_count = 0
    total_count = len(test_prompts)
    
    for i, prompt in enumerate(test_prompts, 1):
        result = greedy_generate(model, prompt, char2idx, idx2char, max_new_tokens=40, device=device)
        print(f"{i:2d}. Input: '{prompt:15}' → Output: '{result}'")
        
        # Check if it's a perfect completion (contains a complete phrase)
        if '.' in result and len(result) > len(prompt) + 10:
            perfect_count += 1
    
    print("=" * 80)
    print(f"📊 RESULTS: {perfect_count}/{total_count} prompts generated complete phrases")
    print(f"Success Rate: {perfect_count/total_count*100:.1f}%")

if __name__ == "__main__":
    main()