#!/usr/bin/env python3
"""Interactive text generation script."""

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
    device = torch.device('cpu')
    
    # Load model
    print("🤖 Loading Transformer model...")
    checkpoint_path = "checkpoints/checkpoint_5000.pt"
    try:
        checkpoint = load_checkpoint(checkpoint_path, device)
        model = create_model_from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Get vocab
        char2idx = checkpoint['config']['vocab_info']['char2idx']
        idx2char = {int(k): v for k, v in checkpoint['config']['vocab_info']['idx2char'].items()}
        
        print(" Model loaded successfully!")
        print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Vocabulary size: {len(char2idx)} characters")
        
    except FileNotFoundError:
        print(f" Checkpoint not found: {checkpoint_path}")
        print("Please make sure the checkpoint file exists.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\n" + "="*60)
    print(" INTERACTIVE TEXT GENERATION")
    print("="*60)
    print("Enter a prompt and the model will complete it!")
    print("Examples: 'Better', 'bird', 'The early', 'journey', 'roads'")
    print("Type 'quit' or 'exit' to stop.")
    print("="*60)
    
    while True:
        try:
            # Get user input
            prompt = input("\n Enter your prompt: ").strip()
            
            # Check for exit conditions
            if prompt.lower() in ['quit', 'exit', 'q']:
                
                break
            
            # Handle empty input
            if not prompt:
                print(" Please enter a prompt!")
                continue
            
            # Generate text
            print("🔄 Generating...")
            result = greedy_generate(
                model, prompt, char2idx, idx2char, 
                max_new_tokens=50, device=device
            )
            
            # Display result
            print("\n" + "="*50)
            print("RESULT:")
            print("="*50)
            print(f"Input:  '{prompt}'")
            print(f"✨ Output: '{result}'")
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n\n Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"Error during generation: {e}")
            print("Please try again with a different prompt.")


if __name__ == "__main__":
    main()