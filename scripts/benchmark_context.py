
import torch
import yaml
import sys
import os
import gc
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indra.models.quantum_core import IndraQuantum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def benchmark_context(config_path='configs/quantum_6gb.yaml'):
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Benchmarking on CPU is not representative of VRAM limits.")
        return

    device = 'cuda'
    config = load_config(config_path)
    
    # Test sequence lengths
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    max_successful_len = 0
    
    logger.info(f"Benchmarking context window for config: {config_path}")
    logger.info(f"Model Config: d_model={config['d_model']}, layers={config['n_layers']}, heads={config['n_heads']}")
    
    for seq_len in seq_lengths:
        logger.info(f"Testing sequence length: {seq_len}")
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Initialize model
            model = IndraQuantum(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                n_layers=config['n_layers'],
                n_heads=config['n_heads'],
                dropout=config['dropout'],
                max_seq_length=seq_len # Update max seq len for this test
            ).to(device)
            
            # Create dummy input
            batch_size = config['batch_size']
            input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)
            
            # Forward pass
            outputs = model(input_ids)
            
            # Backward pass (simulate training)
            loss = outputs.mean()
            loss.backward()
            
            max_successful_len = seq_len
            logger.info(f"SUCCESS: Sequence length {seq_len} fits in VRAM.")
            
            # Clean up
            del model, input_ids, outputs, loss
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.info(f"OOM: Sequence length {seq_len} is too large.")
                break
            else:
                logger.error(f"Error testing length {seq_len}: {e}")
                break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
            
    print("\n" + "="*50)
    print(f"MAXIMUM STABLE CONTEXT WINDOW: {max_successful_len}")
    print("="*50 + "\n")

if __name__ == "__main__":
    benchmark_context()
