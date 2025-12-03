
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def setup_teacher_model(model_name="microsoft/phi-2", cache_dir="./teacher_models"):
    """
    Downloads and sets up a teacher model for Knowledge Distillation.
    Recommended models for local 6GB VRAM:
    - microsoft/phi-2 (2.7B) - Might be tight, quantization recommended
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B) - Very safe for 6GB
    - Qwen/Qwen1.5-1.8B (1.8B) - Good balance
    """
    print(f"Downloading/Loading teacher model: {model_name}...")
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Load model with optimization for low VRAM if possible
        # We use torch_dtype=torch.float16 for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    except ImportError:
        print("Accelerate not found. Loading without low_cpu_mem_usage optimization.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    except Exception as e:
        if "accelerate" in str(e):
             print("Accelerate library missing. Loading without low_cpu_mem_usage.")
             model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            raise e
        
    print(f"Successfully loaded {model_name}")
    print(f"Model parameters: {model.num_parameters() / 1e9:.2f}B")
    
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Teacher Model for KD")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="HuggingFace model ID (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--dir", type=str, default="./teacher_models",
                        help="Directory to save the model")
    
    args = parser.parse_args()
    
    setup_teacher_model(args.model, args.dir)
