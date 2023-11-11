# -*- coding: utf-8 -*-

from transformers import (
    LlamaTokenizer,
    LlamaConfig,
)

from models.llama_lm import LlamaLM
from models.llama_instruct import LlamaInstruct
from models.sparsellama_lm import SparseLlamaLM


def get_model_class(model_type):
    if model_type == "llama_lm":
        tokenizer_class = LlamaTokenizer
        config_class = LlamaConfig
        model_class = LlamaLM
    elif model_type == "llama_instruct":
        tokenizer_class = LlamaTokenizer
        config_class = LlamaConfig
        model_class = LlamaInstruct
    elif model_type == "sparsellama_lm":
        tokenizer_class = LlamaTokenizer
        config_class = LlamaConfig
        model_class = SparseLlamaLM
    else:
        raise KeyError(f"Unknown model type {model_type}.")

    return tokenizer_class, config_class, model_class