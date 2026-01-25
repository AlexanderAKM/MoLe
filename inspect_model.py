# Quick script to inspect ChemBERTa-77M-MLM architecture
from transformers import RobertaModel, RobertaConfig

model_name = "DeepChem/ChemBERTa-77M-MLM"
config = RobertaConfig.from_pretrained(model_name)

print("=" * 60)
print("ChemBERTa-77M-MLM Configuration")
print("=" * 60)

print(f"\n--- Core Architecture ---")
print(f"Hidden size (d_model):        {config.hidden_size}")
print(f"Number of layers:             {config.num_hidden_layers}")
print(f"Number of attention heads:    {config.num_attention_heads}")
print(f"Head dimension:               {config.hidden_size // config.num_attention_heads}")
print(f"Intermediate (FFN) size:      {config.intermediate_size}")
print(f"Vocab size:                   {config.vocab_size}")
print(f"Max position embeddings:      {config.max_position_embeddings}")

print(f"\n--- Per-Layer Component Counts ---")
n_heads = config.num_attention_heads
d_model = config.hidden_size
d_head = d_model // n_heads
d_ffn = config.intermediate_size

print(f"Attention heads per layer:    {n_heads}")
print(f"FFN neurons per layer:        {d_ffn}")

print(f"\n--- Parameter Counts (per layer) ---")
# Attention: Q, K, V, O projections
attn_params_per_head = 4 * d_model * d_head  # Q, K, V, O each: d_model -> d_head or d_head -> d_model
attn_params_total = 4 * d_model * d_model + 4 * d_model  # W_Q, W_K, W_V, W_O + biases
print(f"Attention params (all heads): {attn_params_total:,}")
print(f"Attention params per head:    ~{attn_params_total // n_heads:,}")

# FFN: two linear layers
ffn_params = d_model * d_ffn + d_ffn + d_ffn * d_model + d_model  # W_in, b_in, W_out, b_out
print(f"FFN params:                   {ffn_params:,}")
print(f"FFN params per neuron:        ~{ffn_params // d_ffn:,}")

print(f"\n--- Total Model ---")
model = RobertaModel.from_pretrained(model_name)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters:             {total_params:,}")

# Show layer structure
print(f"\n--- Layer Structure ---")
for name, param in model.named_parameters():
    if "layer.0" in name:  # Just show first layer as example
        print(f"{name}: {param.shape}")
