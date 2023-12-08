# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.modeling_sparsellama
from modules.modeling_sparsellama import SparseLlamaModel, SparseLlamaAttention, SparseLlamaForCausalLM
from modules.flash_attn_monkey_patch_sparsellama import _prepare_decoder_attention_mask, forward
from modules.modeling_sparsellama import SparseLlamaForCausalLM as CustomizedLlamaForCausalLM


SparseLlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
SparseLlamaAttention.forward = forward
SparseLlamaForCausalLM.forward = CustomizedLlamaForCausalLM.forward

import collections


SparseLlamaLM = SparseLlamaForCausalLM
