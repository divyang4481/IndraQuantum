import torch
import sys
import os
import argparse
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from indra.models.quantum_core import IndraQuantum

def chat(checkpoint_path, tt_rank=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Init Model
    vocab_size = 32000
    d_model = 128
    config = {'tt_rank': tt_rank, 'use_tt_embeddings': True}
    model = IndraQuantum(vocab_size, d_model, config=config).to(device)
    
    # Load Weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    print("\n=== IndraQuantum Chat (Type 'exit' to quit) ===")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            prompt = f"User: {user_input}\nAssistant:"
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            
            generated = input_ids
            # Generate
            for _ in range(50):
                with torch.no_grad():
                    logits = model(generated)
                    next_token_logits = logits[:, -1, :]
                    # Top-k sampling
                    top_k = 50
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                    next_token_idx = torch.multinomial(top_k_probs, 1)
                    next_token = torch.gather(top_k_indices, -1, next_token_idx)
                    
                    # Debug: Print top choices
                    # if _ == 0:
                    #     print("\nTop 5 predictions:")
                    #     for i in range(5):
                    #         tok = tokenizer.decode([top_k_indices[0, i].item()])
                    #         prob = top_k_probs[0, i].item()
                    #         print(f"  '{tok}': {prob:.4f}")

                    
                    generated = torch.cat([generated, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            output = tokenizer.decode(generated[0], skip_special_tokens=True)
            # Extract just the assistant response
            if "Assistant:" in output:
                response = output.split("Assistant:")[-1].strip()
            else:
                response = output
            print(f"Indra: {response}\n")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--rank", type=int, default=32, help="TT Rank used in training")
    args = parser.parse_args()
    
    chat(args.checkpoint, args.rank)
