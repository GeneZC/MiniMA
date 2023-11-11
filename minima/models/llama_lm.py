# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.modeling_llama import LlamaModel, LlamaAttention, LlamaForCausalLM
from modules.flash_attn_monkey_patch_llama import _prepare_decoder_attention_mask, forward as attention_forward
from modules.modeling_llama import LlamaForCausalLM as CustomizedLlamaForCausalLM

import modules.modeling_llama


LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
LlamaAttention.forward = attention_forward
LlamaForCausalLM.forward = CustomizedLlamaForCausalLM.forward

import collections


LlamaLM = LlamaForCausalLM
