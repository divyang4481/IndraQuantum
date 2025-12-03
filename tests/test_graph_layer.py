import torch
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indra.models.quantum_graph import ComplexGraphAttentionLayer, IndraQuantumGraph
from indra.graph.builder import TextGraphBuilder
from transformers import AutoTokenizer

def test_layer():
    print("Testing ComplexGraphAttentionLayer...")
    d_model = 64
    n_heads = 4
    layer = ComplexGraphAttentionLayer(d_model, n_heads)
    
    batch_size = 2
    seq_len = 10
    
    # Input: [B, S, D*2]
    x = torch.randn(batch_size, seq_len, d_model * 2)
    
    # Graph Mask: [B, S, S] (Edge Types 0-3)
    mask = torch.randint(0, 4, (batch_size, seq_len, seq_len)).float()
    
    out = layer(x, mask)
    print("Output shape:", out.shape)
    assert out.shape == (batch_size, seq_len, d_model * 2)
    print("Layer test passed.")

def test_builder_and_model():
    print("\nTesting Builder and Model...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    builder = TextGraphBuilder(tokenizer, max_seq_len=32)
    
    text = "Hello world. This is a test."
    graph = builder.build_graph(text)
    
    input_ids = graph['input_ids'].unsqueeze(0) # [1, S]
    node_types = graph['node_types'].unsqueeze(0) # [1, S]
    graph_mask = graph['graph_mask'].unsqueeze(0) # [1, S, S]
    
    print("Input IDs:", input_ids.shape)
    print("Node Types:", node_types.shape)
    print("Graph Mask:", graph_mask.shape)
    
    # Check special tokens
    print("Special Tokens in Input:", input_ids[input_ids >= 32000])
    
    # Model
    vocab_size = 32002
    d_model = 32
    model = IndraQuantumGraph(vocab_size, d_model, n_layers=2, n_heads=4)
    
    logits = model(input_ids, node_types, graph_mask)
    print("Logits shape:", logits.shape)
    assert logits.shape == (1, input_ids.size(1), vocab_size)
    print("Model test passed.")

if __name__ == "__main__":
    test_layer()
    test_builder_and_model()
