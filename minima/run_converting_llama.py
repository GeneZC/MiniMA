
import argparse

import torch

from modules.modeling_sparsellama import SparseLlamaForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument('--sparsellama_dir', default="sparsellama_lm-7b", type=str)
args = parser.parse_args()

m = SparseLlamaForCausalLM.from_pretrained(args.sparsellama_dir, torch_dtype=torch.float16)

num_hidden_layers = m.config.num_hidden_layers
hidden_size = m.config.hidden_size
num_attention_heads = m.config.num_attention_heads
intermediate_size = m.config.intermediate_size

# Prune hiddens, heads, neurons.
sparsity_map = {
    "3b": {
        "hidden": {str(i - 1): hidden_size - 3072 for i in range(num_hidden_layers + 1)},
        "head": {str(i): num_attention_heads - 24 for i in range(num_hidden_layers)},
        "neuron": {str(i): intermediate_size - 8192 for i in range(num_hidden_layers)}
    }
}
m.config.sparsity = "3b"
m.config.sparsity_map = sparsity_map
m.model.sparsify("3b")
m.model.densify()

indices = m.model.recover_indices.long()[:3072]
weight = m.lm_head.weight.index_select(1, indices).clone().detach()
m.lm_head.weight = torch.nn.Parameter(torch.empty_like(weight))
m.lm_head.weight.requires_grad = False
m.lm_head.weight.copy_(weight.contiguous())
m.lm_head.weight.requires_grad = True

# Prune layers.
d = int((num_hidden_layers - 24) / 2) # Should be divisible.
m.model.layers = torch.nn.ModuleList(m.model.layers[d: -d]) # Is this behavior safe?

m.config.hidden_size = 3072
m.config.num_hidden_layers = 24
m.config.num_attention_heads = 24
m.config.intermediate_size = 8192

m.save_pretrained("minima-3b-init")