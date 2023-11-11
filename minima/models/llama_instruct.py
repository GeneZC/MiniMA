# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers.models.llama.modeling_llama
from modules.flash_attn_monkey_patch_llama import _prepare_decoder_attention_mask, forward
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, LlamaAttention

LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
LlamaAttention.forward = forward

import collections


class LlamaInstruct(LlamaPreTrainedModel):
    Output = collections.namedtuple(
        "Output", 
        (
            'logits',
            'labels',
        )
    )

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

    def forward(self, inputs):
        text_indices, text_mask, labels, label_mask = inputs

        hidden_states = self.model(text_indices, attention_mask=text_mask)[0]
        logits = self.lm_head(hidden_states)
        
        logit_size = logits.shape[-1]
        # mask = label_mask.unsqueeze(-1).expand_as(logits)
        # logits = torch.masked_select(logits, mask)
        logits = logits.reshape(-1, logit_size)
        # mask = label_mask
        # labels = torch.masked_select(labels, mask)
        labels[~label_mask] = -100 # ignore_index.
        labels = labels.reshape(-1)

        # predictions = logits.argmax(-1)

        return LlamaInstruct.Output(
            logits=logits, 
            labels=labels,
        )._asdict()

    @staticmethod
    def loss_fn(output):
        loss = F.cross_entropy(output.logits, output.labels, reduction="mean")
        return loss 