import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import os
import warnings

def load_llava_13b(quantization="8bit", use_flash_attn=None):
    """
    Load LLaVA 13B model with specified configuration
    
    Args:
        quantization: "none", "8bit", or "4bit"
        use_flash_attn: None (auto-detect), True, or False
    """
    
    # Auto-detect flash attention availability
    if use_flash_attn is None:
        try:
            import flash_attn
            use_flash_attn = True
            print("Flash attention detected and will be used")
        except ImportError as e:
            use_flash_attn = False
            print(f"Flash attention not available: {e}")
            print("Continuing without flash attention (slightly slower but functional)")
    
    model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
    
    print(f"Loading model: {model_id}")
    print(f"Quantization: {quantization}")
    print(f"Flash attention: {use_flash_attn}")
    
    # Load processor
    print("Loading processor...")
    try:
        processor = LlavaNextProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading processor: {e}")
        # Try with authentication token
        token = os.getenv('HF_TOKEN')
        if token:
            processor = LlavaNextProcessor.from_pretrained(model_id, use_auth_token=token)
        else:
            print("Please set HF_TOKEN environment variable or run 'huggingface-cli login'")
            raise
    
    # Configure model loading parameters
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    
    # Disable flash attention if not available
    if not use_flash_attn:
        model_kwargs["attn_implementation"] = "eager"
    
    # Add quantization settings
    if quantization == "8bit":
        model_kwargs["load_in_8bit"] = True
    elif quantization == "4bit":
        model_kwargs["load_in_4bit"] = True
        model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
        model_kwargs["bnb_4bit_use_double_quant"] = True
        model_kwargs["bnb_4bit_quant_type"] = "nf4"
    
    # Load model
    print("Loading model (this may take several minutes)...")
    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, **model_kwargs
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try with authentication token
        token = os.getenv('HF_TOKEN')
        if token:
            model_kwargs["use_auth_token"] = token
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, **model_kwargs
            )
        else:
            raise
    
    print("Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    
    # Test basic functionality
    print("Testing model...")
    test_input = torch.randint(0, 1000, (1, 10)).to(model.device)
    with torch.no_grad():
        try:
            _ = model(input_ids=test_input)
            print("Model test passed!")
        except Exception as e:
            print(f"Model test failed: {e}")
    
    return model, processor

def check_system_compatibility():
    """Check system compatibility for LLaVA 13B deployment"""
    print("=== LLaVA 13B System Compatibility Check ===")
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # Check PyTorch
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory // 1024**3
            print(f"  GPU {i}: {props.name} ({memory_gb} GB)")
            
            # Check if GPU has enough memory for 13B model
            if memory_gb >= 24:
                print(f"    ✓ Sufficient memory for 13B model")
            elif memory_gb >= 13:
                print(f"    ⚠ Marginal memory - use 8-bit or 4-bit quantization")
            else:
                print(f"    ✗ Insufficient memory for 13B model - consider 7B variant")
    else:
        print("✗ CUDA not available - GPU required for 13B model")
    
    # Check GLIBC version
    try:
        import platform
        print(f"Platform: {platform.platform()}")
        
        # Try to check GLIBC version
        import subprocess
        result = subprocess.run(['ldd', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            glibc_version = result.stdout.split('\n')[0]
            print(f"GLIBC: {glibc_version}")
        else:
            print("GLIBC version check failed")
    except Exception as e:
        print(f"System check error: {e}")
    
    # Check flash attention
    try:
        import flash_attn
        print(f"Flash attention: Available (version {flash_attn.__version__})")
    except ImportError:
        print("Flash attention: Not available (will use eager attention)")
    
    # Check transformers version
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")
    
    # Check other dependencies
    try:
        import bitsandbytes
        print(f"BitsAndBytes: Available (version {bitsandbytes.__version__})")
    except ImportError:
        print("BitsAndBytes: Not available (required for quantization)")
    
    print("=== End Compatibility Check ===\n")

if __name__ == "__main__":
    # Run system check first
    check_system_compatibility()
    
    # Load 13B model
    try:
        print("Loading LLaVA 13B model with 8-bit quantization...")
        model, processor = load_llava_13b(quantization="8bit")
        print("✓ LLaVA 13B model loaded successfully!")
        
        # Memory usage info
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
    except Exception as e:
        print(f"✗ LLaVA 13B model loading failed: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Try 4-bit quantization: load_llava_13b(quantization='4bit')")
        print("2. Check GPU memory availability")
        print("3. Ensure HuggingFace authentication is set up")
        print("4. Consider using the 7B model variant if memory is limited")