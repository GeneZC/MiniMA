# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch SparseLLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaPreTrainedModel


logger = logging.get_logger(__name__)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class SparseLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, sparsified_elements=0):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.normalized_shape_origin = (hidden_size,)
        self.normalized_shape_sparsified = (hidden_size,)
        self.sparsify(sparsified_elements)
        self.densify()

    def forward(self, hidden_states, hidden_mask=None):
        weight = self.weight[:self.normalized_shape_sparsified[0]]

        if hidden_mask is not None:
            remain_indices = torch.where(~hidden_mask.squeeze().eq(0))[0]
            _hidden_states = hidden_states.clone().to(torch.float32)
            hidden_states = hidden_states.index_select(-1, remain_indices)
            weight = weight[remain_indices]
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            _hidden_states[:, :, remain_indices] = hidden_states
            hidden_states = _hidden_states
        else:
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return weight * hidden_states

    def reorder(self, indices):
        indices = indices.to(self.weight.device)
        weight = self.weight[indices].clone().detach()
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True

    def sparsify(self, num_elements):
        self.normalized_shape_sparsified = (self.normalized_shape_origin[0] - round_to_multiple_of_eight(num_elements),)

    def densify(self):
        weight = self.weight[:self.normalized_shape_sparsified[0]].clone().detach()
        self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True
        self.normalized_shape = self.normalized_shape_sparsified


class SparseLlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def round_to_multiple_of_eight(input_size):
    return round(input_size * 1.0 / 8) * 8


class SparseEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, sparsified_elements=0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim_origin = embedding_dim
        self.embedding_dim_sparsified = embedding_dim
        self.sparsify(sparsified_elements)
        self.densify()

    def forward(self, input):
        weight = self.weight[:, :self.embedding_dim_sparsified]
        #return F.linear(
        return F.embedding(input, weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse)
        #        vh.t(),
        #        False
        #    )

    def reorder(self, indices):
        indices = indices.to(self.weight.device)
        weight = self.weight.index_select(1, indices).clone().detach()
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True

    def sparsify(self, num_elements):
        self.embedding_dim_sparsified = self.embedding_dim_origin - round_to_multiple_of_eight(num_elements)

    def densify(self):
        self.embedding_dim = self.embedding_dim_sparsified
        weight = self.weight[:, :self.embedding_dim_sparsified].clone().detach()
        self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, element_size=1, dim=0, sparsified_elements=(0, 0)):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features_origin = in_features
        self.out_features_origin = out_features
        self.in_features_sparsified = in_features
        self.out_features_sparsified = out_features
        self.element_size = element_size
        self.dim = dim
        self.sparsify(sparsified_elements[0])
        self.sparsify(sparsified_elements[1], for_hidden=True)
        self.densify()

    def forward(self, input):
        weight = self.weight[:self.out_features_sparsified, :self.in_features_sparsified]
        if self.bias is not None:
            bias = self.bias[:self.out_features_sparsified]
        else:
            bias = self.bias
        return F.linear(input, weight, bias)

    def reorder(self, indices, for_hidden=False):
        if for_hidden:
            self.dim = 1 - self.dim

        indices = indices.to(self.weight.device)
        weight = self.weight.index_select(1 - self.dim, indices).clone().detach()
        if self.bias is not None:
            if self.dim == 0:
                bias = self.bias.clone().detach()
            else:
                bias = self.bias[indices].clone().detach()
        #self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True
        if self.bias is not None:
            #self.bias = nn.Parameter(torch.empty_like(bias))
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True

        if for_hidden:
            self.dim = 1 - self.dim

    def sparsify(self, num_elements, for_hidden=False):
        if for_hidden:
            self.dim = 1 - self.dim
            cache_element_size = self.element_size
            self.element_size = 1

        if self.dim == 0:
            self.in_features_sparsified = self.in_features_origin - round_to_multiple_of_eight(num_elements * self.element_size)
        if self.dim == 1:
            self.out_features_sparsified = self.out_features_origin - round_to_multiple_of_eight(num_elements * self.element_size)

        if for_hidden:
            self.dim = 1 - self.dim
            self.element_size = cache_element_size

    def densify(self):
        self.in_features = self.in_features_sparsified
        self.out_features = self.out_features_sparsified
        weight = self.weight[:self.out_features_sparsified, :self.in_features_sparsified].clone().detach()
        if self.bias is not None:
            bias = self.bias[:self.out_features_sparsified].clone().detach()
        # else:
        #     bias = self.bias.clone().detach()
        self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty_like(bias))
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True


class SparseLlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        sparsified_neurons,
        sparsified_hiddens,
    ):
        super().__init__()
        self.gate_proj = SparseLinear(hidden_size, intermediate_size, bias=False, element_size=1, dim=1, sparsified_elements=(sparsified_neurons, sparsified_hiddens))
        self.down_proj = SparseLinear(intermediate_size, hidden_size, bias=False, element_size=1, dim=0, sparsified_elements=(sparsified_neurons, sparsified_hiddens))
        self.up_proj = SparseLinear(hidden_size, intermediate_size, bias=False, element_size=1, dim=1, sparsified_elements=(sparsified_neurons, sparsified_hiddens))
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x, neuron_mask=None):
        if neuron_mask is not None:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x) * neuron_mask)
        else:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SparseLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, sparsified_heads, sparsified_hiddens):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.hidden_size_sparsified = self.hidden_size
        self.num_heads_sparsified = self.num_heads

        self.q_proj = SparseLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, element_size=self.head_dim, dim=1, sparsified_elements=(sparsified_heads, sparsified_hiddens))
        self.k_proj = SparseLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, element_size=self.head_dim, dim=1, sparsified_elements=(sparsified_heads, sparsified_hiddens))
        self.v_proj = SparseLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, element_size=self.head_dim, dim=1, sparsified_elements=(sparsified_heads, sparsified_hiddens))
        self.o_proj = SparseLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, element_size=self.head_dim, dim=0, sparsified_elements=(sparsified_heads, sparsified_hiddens))
        self.rotary_emb = SparseLlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads_sparsified, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        self.hidden_size_sparsified = self.o_proj.in_features_sparsified
        self.num_heads_sparsified = int(self.hidden_size_sparsified / self.head_dim)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads_sparsified, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads_sparsified, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads_sparsified, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads_sparsified, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads_sparsified, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads_sparsified, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads_sparsified, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size_sparsified)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class SparseLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.sparsified_heads = config.sparsity_map[config.sparsity]["head"].get(str(layer_idx), 0)
        self.sparsified_neurons = config.sparsity_map[config.sparsity]["neuron"].get(str(layer_idx), 0)
        self.sparsified_hiddens = config.sparsity_map[config.sparsity]["hidden"].get(str(layer_idx), 0)
        self.self_attn = SparseLlamaAttention(config=config, sparsified_heads=self.sparsified_heads, sparsified_hiddens=self.sparsified_hiddens)
        self.mlp = SparseLlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            sparsified_neurons=self.sparsified_neurons,
            sparsified_hiddens=self.sparsified_hiddens,
        )
        self.input_layernorm = SparseLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, sparsified_elements=self.sparsified_hiddens)
        self.post_attention_layernorm = SparseLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, sparsified_elements=self.sparsified_hiddens)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask=None,
        neuron_mask=None,
        hidden_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if self.self_attn.o_proj.in_features_sparsified >= 8:
            # Self Attention
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states, hidden_mask=hidden_mask)
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            if hidden_mask is not None:
                hidden_states = hidden_states * hidden_mask
            hidden_states = residual + hidden_states
            if hidden_mask is not None:
                hidden_states = hidden_states * hidden_mask
        else:
            pass
        
        if self.mlp.down_proj.in_features_sparsified >= 8:
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states, hidden_mask=hidden_mask)
            hidden_states = self.mlp(hidden_states, neuron_mask=neuron_mask)
            if hidden_mask is not None:
                hidden_states = hidden_states * hidden_mask
            hidden_states = residual + hidden_states
            if hidden_mask is not None:
                hidden_states = hidden_states * hidden_mask
        else:
            pass

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class SparseLlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`SparseLlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config

        if not hasattr(self.config, "sparsity"):
            self.config.sparsity = "0"
        if not hasattr(self.config, "sparsity_map"):
            self.config.sparsity_map = {"0": {"hidden": {}, "head": {}, "neuron": {}}}

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.sparsified_hiddens = config.sparsity_map[config.sparsity]["hidden"].get("-1", 0)
        self.embed_tokens = SparseEmbedding(config.vocab_size, config.hidden_size, self.padding_idx, sparsified_elements=self.sparsified_hiddens)
        self.layers = nn.ModuleList([SparseLlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = SparseLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, sparsified_elements=self.sparsified_hiddens)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        # self.sparsify(self.config.sparsity)
        # self.densify()
        self.recover_indices = nn.Parameter(torch.arange(self.config.hidden_size).float())

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask=None,
        neuron_mask=None,
        hidden_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if hidden_mask is not None:
            inputs_embeds = inputs_embeds * hidden_mask

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        # head_mask has shape [num_hidden_layers x batch_size (*1) x num_heads x seq_length (*1)  x head_size (*1)]
        # neuron_mask has shape [num_hidden_layers x batch_size (*1) x seq_length (*1) x intermediate_size]
        # hidden_mask has shape [batch_size (*1) x seq_length (*1) x hidden_size]
        if head_mask is not None:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(self.dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        if neuron_mask is not None:
            neuron_mask = neuron_mask.unsqueeze(1).unsqueeze(1).to(self.dtype)
        else:
            neuron_mask = [None] * self.config.num_hidden_layers
        if hidden_mask is not None:
            hidden_mask = hidden_mask.unsqueeze(0).unsqueeze(1).to(self.dtype)
        else:
            hidden_mask = None

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    head_mask[idx],
                    neuron_mask[idx],
                    hidden_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[idx],
                    neuron_mask=neuron_mask[idx],
                    hidden_mask=hidden_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states, hidden_mask=hidden_mask)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Recover the full hidden size for later use.
        hidden_states_shape = hidden_states.shape[:-1] + (self.config.hidden_size,)
        _hidden_states = torch.zeros(*hidden_states_shape).to(hidden_states.device).type(hidden_states.dtype)
        _hidden_states.index_copy_(2, self.recover_indices.long().clamp(max=self.config.hidden_size - 1)[:hidden_states.shape[-1]], hidden_states)
        hidden_states = _hidden_states

        next_cache = next_decoder_cache if use_cache else None
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    def reorder(self, head_indices, neuron_indices, hidden_indices):
        for layer_idx, indices in head_indices.items():
            n, h = self.layers[layer_idx].self_attn.num_heads, self.layers[layer_idx].self_attn.head_dim
            indices = torch.arange(n * h).reshape(n, h)[indices.cpu()].reshape(-1).contiguous().long()
            self.layers[layer_idx].self_attn.q_proj.reorder(indices)
            self.layers[layer_idx].self_attn.k_proj.reorder(indices)
            self.layers[layer_idx].self_attn.v_proj.reorder(indices)
            self.layers[layer_idx].self_attn.o_proj.reorder(indices)
        for layer_idx, indices in neuron_indices.items():
            self.layers[layer_idx].mlp.up_proj.reorder(indices)
            self.layers[layer_idx].mlp.gate_proj.reorder(indices)
            self.layers[layer_idx].mlp.down_proj.reorder(indices)
        for layer_idx, indices in hidden_indices.items():
            if layer_idx == -1:
                self.embed_tokens.reorder(indices)
                self.norm.reorder(indices)
                recover_indices = indices.to(self.recover_indices.device)
                self.recover_indices.requires_grad = False
                self.recover_indices.copy_(recover_indices.contiguous())
                self.recover_indices.requires_grad = True
            else:
                self.layers[layer_idx].input_layernorm.reorder(indices)
                self.layers[layer_idx].self_attn.q_proj.reorder(indices, for_hidden=True)
                self.layers[layer_idx].self_attn.k_proj.reorder(indices, for_hidden=True)
                self.layers[layer_idx].self_attn.v_proj.reorder(indices, for_hidden=True)
                self.layers[layer_idx].self_attn.o_proj.reorder(indices, for_hidden=True)
                self.layers[layer_idx].post_attention_layernorm.reorder(indices)
                self.layers[layer_idx].mlp.up_proj.reorder(indices, for_hidden=True)
                self.layers[layer_idx].mlp.gate_proj.reorder(indices, for_hidden=True)
                self.layers[layer_idx].mlp.down_proj.reorder(indices, for_hidden=True)

    def sparsify(self, sparsity):
        assert sparsity in self.config.sparsity_map, f"Sparsity {sparsity} is not in the sparsity map {self.config.sparsity_map}."
        #self.sparsity = sparsity
        head_map, neuron_map, hidden_map = \
            self.config.sparsity_map[sparsity]["head"], self.config.sparsity_map[sparsity]["neuron"], self.config.sparsity_map[sparsity]["hidden"]
        
        for layer_idx in range(self.config.num_hidden_layers):
            self.layers[layer_idx].self_attn.q_proj.sparsify(head_map.get(str(layer_idx), 0))
            self.layers[layer_idx].self_attn.k_proj.sparsify(head_map.get(str(layer_idx), 0))
            self.layers[layer_idx].self_attn.v_proj.sparsify(head_map.get(str(layer_idx), 0))
            self.layers[layer_idx].self_attn.o_proj.sparsify(head_map.get(str(layer_idx), 0))
            self.layers[layer_idx].mlp.up_proj.sparsify(neuron_map.get(str(layer_idx), 0))
            self.layers[layer_idx].mlp.gate_proj.sparsify(neuron_map.get(str(layer_idx), 0))
            self.layers[layer_idx].mlp.down_proj.sparsify(neuron_map.get(str(layer_idx), 0))
            self.layers[layer_idx].input_layernorm.sparsify(hidden_map.get(str(layer_idx), 0))
            self.layers[layer_idx].self_attn.q_proj.sparsify(hidden_map.get(str(layer_idx), 0), for_hidden=True)
            self.layers[layer_idx].self_attn.k_proj.sparsify(hidden_map.get(str(layer_idx), 0), for_hidden=True)
            self.layers[layer_idx].self_attn.v_proj.sparsify(hidden_map.get(str(layer_idx), 0), for_hidden=True)
            self.layers[layer_idx].self_attn.o_proj.sparsify(hidden_map.get(str(layer_idx), 0), for_hidden=True)
            self.layers[layer_idx].post_attention_layernorm.sparsify(hidden_map.get(str(layer_idx), 0))
            self.layers[layer_idx].mlp.up_proj.sparsify(hidden_map.get(str(layer_idx), 0), for_hidden=True)
            self.layers[layer_idx].mlp.gate_proj.sparsify(hidden_map.get(str(layer_idx), 0), for_hidden=True)
            self.layers[layer_idx].mlp.down_proj.sparsify(hidden_map.get(str(layer_idx), 0), for_hidden=True)
        self.embed_tokens.sparsify(hidden_map.get("-1", 0))
        self.norm.sparsify(hidden_map.get("-1", 0))

    def densify(self):
        for layer_idx in range(self.config.num_hidden_layers):
            self.layers[layer_idx].input_layernorm.densify()
            self.layers[layer_idx].self_attn.q_proj.densify()
            self.layers[layer_idx].self_attn.k_proj.densify()
            self.layers[layer_idx].self_attn.v_proj.densify()
            self.layers[layer_idx].self_attn.o_proj.densify()
            self.layers[layer_idx].post_attention_layernorm.densify()
            self.layers[layer_idx].mlp.up_proj.densify()
            self.layers[layer_idx].mlp.gate_proj.densify()
            self.layers[layer_idx].mlp.down_proj.densify()
        self.embed_tokens.densify()
        self.norm.densify()


class SparseLlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = SparseLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask=None,
        neuron_mask=None,
        hidden_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            neuron_mask=neuron_mask,
            hidden_mask=hidden_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # # Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # NOTE: Our version does not need shift since they have already been shifted.
            shift_logits = logits
            shift_labels = labels
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
            # NOTE: Our version needs the use of mask since we do not use `ignore_index`.
            logit_size = shift_logits.shape[-1]
            mask = attention_mask.unsqueeze(-1).expand_as(shift_logits)
            # shift_logits = torch.masked_select(shift_logits, mask)
            shift_logits = shift_logits.reshape(-1, logit_size)
            mask = attention_mask
            # shift_labels = torch.masked_select(shift_labels, mask)
            shift_labels = shift_labels.reshape(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        output = (shift_logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
