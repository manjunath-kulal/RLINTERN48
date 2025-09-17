# 🔄 How to Reproduce Our Results

Follow these simple steps to reproduce the Mini Transformer results:

## 📋 Prerequisites

```bash
# Make sure you have Python 3.8+ installed
python --version

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Step-by-Step Reproduction

### 1. **Run Tests** (Optional but Recommended)
```bash
# Make sure everything works
python -m pytest tests/ -v
```

### 2. **Train the Model**
```bash
# Train with the exact same settings
python experiments/train_charlm.py \
    --data data/tiny.txt \
    --seq_len 32 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --lr 1e-3 \
    --batch_size 16 \
    --steps 500 \
    --eval_interval 100
```

**Expected Output:**
- Training completes in ~6 seconds
- Final BPC: ~0.24 (should be ≤ 1.35)
- Model saved to `checkpoints/best.pt`

### 3. **Evaluate the Model**
```bash
# Get detailed metrics
python experiments/eval.py --ckpt checkpoints/best.pt --data data/tiny.txt
```

**Expected Results:**
- BPC: ~0.24
- Perplexity: ~1.18
- Accuracy: ~93.9%

### 4. **Generate Text**
```bash
# Try different prompts
python experiments/generate.py --ckpt checkpoints/best.pt --prompt "The quick" --max_new_tokens 30

python experiments/generate.py --ckpt checkpoints/best.pt --prompt "To be" --max_new_tokens 20

python experiments/generate.py --ckpt checkpoints/best.pt --prompt "All that glitters" --max_new_tokens 15
```

**Expected Generations:**
- Should complete famous phrases correctly
- Text should be coherent and grammatical

## 🎯 Success Criteria

You've successfully reproduced our results if:
- ✅ BPC ≤ 1.35 (we got 0.24)
- ✅ Model generates coherent text
- ✅ All tests pass
- ✅ Training completes without errors

## 🐛 Troubleshooting

**Training too slow?** Try smaller parameters:
```bash
python experiments/train_charlm.py --data data/tiny.txt --steps 200 --batch_size 8
```

**Out of memory?** Reduce model size:
```bash
python experiments/train_charlm.py --data data/tiny.txt --d_model 64 --n_heads 2
```

**Tests failing?** Make sure you have the right dependencies:
```bash
pip install -r requirements.txt --upgrade
```

That's it! 🎉 You should now have a working Mini Transformer!
