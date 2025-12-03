
import torch
import argparse
import yaml
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indra.models.quantum_core import IndraQuantum
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Test IndraQuantum Checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/quantum_6gb.yaml', help='Path to config file')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Tokenizer name')
    parser.add_argument('--prompt', type=str, default="The quantum cat", help='Input prompt')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    logger.info("Initializing model...")
    model = IndraQuantum(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length']
    )
    
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Handle both full checkpoint and state_dict only
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    
    logger.info(f"Generating text for prompt: '{args.prompt}'")
    input_ids = tokenizer.encode(args.prompt, return_tensors='pt').to(args.device)
    
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=50
    )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(generated_text)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
