# 🖥️ How to Run Mini Transformer on Another PC

Follow these steps to transfer and run the Mini Transformer project on any other computer.

## 📦 Step 1: Transfer the Project

### Option A: Copy the Folder
1. Copy the entire `transformer task` folder to a USB drive or cloud storage
2. Transfer it to the new PC
3. Extract/copy to desired location (e.g., `Desktop/transformer-task/`)

### Option B: Use Git (if you have it)
```bash
# If you've pushed to GitHub, clone it:
git clone https://github.com/yourusername/transformer-task.git
cd transformer-task
```

### Option C: Create Zip File
```bash
# On current PC, create a zip:
zip -r transformer-task.zip "transformer task" -x "*.venv*" "*__pycache__*" "*checkpoints*"

# Transfer the zip file and extract on new PC
```

## 🐍 Step 2: Install Python (if needed)

### Windows:
1. Download Python 3.8+ from https://python.org
2. **Important:** Check "Add Python to PATH" during installation
3. Verify: Open Command Prompt and run `python --version`

### Mac:
```bash
# Install Homebrew first (if not installed):
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python:
brew install python
```

### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

## 🚀 Step 3: Set Up the Environment

Open terminal/command prompt and navigate to the project folder:

```bash
# Navigate to project (adjust path as needed)
cd Desktop/transformer-task

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ✅ Step 4: Verify Installation

Run the tests to make sure everything works:

```bash
python -m pytest tests/ -v
```

**Expected output:** All 36 tests should pass ✅

## 🎯 Step 5: Train Your First Model

```bash
# Train a small model (takes ~10 seconds)
python experiments/train_charlm.py --data data/tiny.txt --steps 200 --batch_size 8

# You should see training progress and final BPC < 1.35
```

## 🎪 Step 6: Generate Text

```bash
# Generate text from your trained model
python experiments/generate.py --ckpt checkpoints/best.pt --prompt "The quick" --max_new_tokens 20
```

## 🐛 Troubleshooting

### Problem: "python not found"
**Solution:** 
- Windows: Try `py` instead of `python`
- Make sure Python is in PATH
- Restart terminal after Python installation

### Problem: "pip not found"
**Solution:**
```bash
# Try these alternatives:
python -m ensurepip --upgrade
# or
python3 -m pip install -r requirements.txt
```

### Problem: PyTorch installation issues
**Solution:**
```bash
# Install CPU-only PyTorch (smaller, faster for this project):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pytest
```

### Problem: Permission errors
**Solution:**
```bash
# Use --user flag:
pip install -r requirements.txt --user
```

### Problem: Virtual environment issues
**Solution:**
```bash
# Skip virtual environment (not recommended but works):
pip install -r requirements.txt
```

## 📋 Quick Test Commands

After setup, test these commands work:

```bash
# 1. Check Python
python --version

# 2. Check dependencies
python -c "import torch, numpy, matplotlib; print('All good!')"

# 3. Run a quick test
python -c "from src.attention_numpy import scaled_dot_product_attention; print('Import works!')"

# 4. Train tiny model
python experiments/train_charlm.py --data data/tiny.txt --steps 50
```

## 🎉 Success!

If all steps work, you should see:
- ✅ Tests pass
- ✅ Training completes
- ✅ Text generation works
- ✅ BPC < 1.35 achieved

Your Mini Transformer is now running on the new PC! 🚀

## 📁 Project Structure Reminder

```
transformer-task/
├── src/                    # Core components
├── experiments/           # Training scripts
├── tests/                # Test suite
├── data/                 # Training data
├── requirements.txt      # Dependencies
├── README.md            # Getting started
└── .venv/               # Virtual environment (auto-created)
```

Need help? Check README.md for basic usage or REPRO.md for detailed reproduction steps!
