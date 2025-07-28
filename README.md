# LLaVA 13B Model - Dependencies & System Information

## 📋 System Configuration

### Environment Details
```
Operating System: Linux-4.18.0-553.36.1.el8_10.x86_64 (RHEL 8)
GLIBC Version: 2.28
Python Version: 3.10.13 (conda-forge)
Conda Environment: llava-env
```

### Hardware Specifications
```
GPU: NVIDIA L40S (44GB VRAM)
Memory Usage: ~13.5GB GPU, ~8GB RAM
CUDA: Available (detected by PyTorch)
```

## 🔧 Dependencies

### Core Dependencies
```yaml
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

python: 3.10.13
pytorch: 2.7.1+cu118
torchvision: [latest compatible]
torchaudio: [latest compatible]
```

### Python Packages (pip)
```
transformers==4.54.0
bitsandbytes==0.46.1
accelerate==latest
huggingface-hub==latest
safetensors==latest
pillow==latest
requests==latest
```

### System Libraries (conda)
```
protobuf (conda-forge)
sentencepiece (conda-forge)
```

## 🚫 Excluded Dependencies

### Flash Attention
```
Status: Not installed (GLIBC 2.28 incompatibility)
Requirement: GLIBC >= 2.32
Alternative: Using eager attention implementation
Performance Impact: ~5-10% slower, fully functional
```

### Deprecated Warnings
```
- BitsAndBytesConfig warning: Using legacy load_in_8bit parameter
- TRANSFORMERS_CACHE warning: Use HF_HOME instead (handled)
- Slow image processor warning: use_fast=False (default behavior)
```

## 📦 Installation Commands

### Complete Environment Setup
```bash
# Create environment
conda create -n llava-env python=3.10 -y
conda activate llava-env

# Install PyTorch (pip method for compatibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install conda packages
conda install -c conda-forge pillow requests protobuf sentencepiece -y

# Install remaining packages
pip install transformers bitsandbytes accelerate huggingface-hub safetensors
```

### Version Verification
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 🔄 Environment Variables

### Required Configuration
```bash
# HuggingFace Authentication
export HF_TOKEN="hf_your_token_here"

# Cache Directory (recommended)
export HF_HOME="/path/to/large/disk/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# Optional: Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Make Permanent
```bash
# Add to ~/.bashrc
echo 'export HF_TOKEN="your_token"' >> ~/.bashrc
echo 'export HF_HOME="/your/cache/path"' >> ~/.bashrc
source ~/.bashrc
```

## 📊 Performance Metrics

### Model Loading
```
Initial Load Time: ~90 seconds (first download)
Cached Load Time: ~8 seconds
Model Size: ~25GB (download), ~13GB (GPU memory)
Quantization: 8-bit (50% memory reduction)
```

### Inference Performance
```
GPU Memory Usage: 12.76GB allocated, 13.50GB reserved
Available Memory: ~30GB remaining on L40S
Generation Speed: ~10-20 tokens/second
Image Processing: ~0.1-0.2 seconds
```

## 🔧 Configuration Files

### environment.yml
```yaml
name: llava-env
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pillow
  - requests
  - protobuf
  - sentencepiece
  - pip
  - pip:
    - torch --index-url https://download.pytorch.org/whl/cu118
    - torchvision --index-url https://download.pytorch.org/whl/cu118
    - torchaudio --index-url https://download.pytorch.org/whl/cu118
    - transformers
    - bitsandbytes
    - accelerate
    - huggingface-hub
    - safetensors
```

### requirements.txt (pip only)
```
torch --index-url https://download.pytorch.org/whl/cu118
torchvision --index-url https://download.pytorch.org/whl/cu118
torchaudio --index-url https://download.pytorch.org/whl/cu118
transformers==4.54.0
bitsandbytes==0.46.1
accelerate
huggingface-hub
safetensors
pillow
requests
protobuf
sentencepiece
```

## ⚠️ Known Issues & Solutions

### 1. Flash Attention GLIBC Error
```
Error: undefined symbol: iJIT_NotifyEvent
Solution: Skip flash-attn installation, use eager attention
Impact: Minimal performance decrease
```

### 2. Disk Space Issues
```
Error: No space left on device (os error 28)
Solution: Set custom HF_HOME to location with >30GB free
Command: export HF_HOME="/path/to/large/disk"
```

### 3. PyTorch Installation Conflicts
```
Error: Various import errors with conda pytorch
Solution: Use pip installation for PyTorch
Reason: Better compatibility with CUDA libraries
```

### 4. Authentication Errors
```
Error: Cannot authenticate through git-credential
Solution: Use huggingface-cli login or HF_TOKEN environment variable
```

## 🔍 Compatibility Matrix

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| Python | 3.10.13 | ✅ Tested | Recommended version |
| PyTorch | 2.7.1+cu118 | ✅ Working | CUDA 11.8 build |
| Transformers | 4.54.0 | ✅ Working | Latest compatible |
| CUDA | 11.8 | ✅ Detected | Hardware compatible |
| Flash Attention | N/A | ❌ Skipped | GLIBC incompatible |
| BitsAndBytes | 0.46.1 | ✅ Working | Quantization support |
| GLIBC | 2.28 | ⚠️ Limited | Flash-attn incompatible |

## 💾 Storage Requirements

| Component | Size | Location |
|-----------|------|----------|
| Model Download | ~25GB | HF_HOME/hub/ |
| Conda Environment | ~8GB | ~/.conda/envs/llava-env/ |
| Model in GPU | ~13GB | GPU VRAM |
| Temporary Files | ~2GB | /tmp/ (during download) |
| **Total Required** | **~48GB** | **Various locations** |

## 🎯 Quick Validation

```bash
# Test complete setup
python -c "
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
print('✅ All imports successful')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

Last Updated: July 27, 2025
