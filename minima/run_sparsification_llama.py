# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import deepspeed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.ops.adam import FusedAdam

import transformers
from transformers.deepspeed import HfDeepSpeedConfig

from tqdm.auto import tqdm

from data import get_pipeline_class, TFRecordReader, TFRecordDataset, TFRecordDistributedDataset
from models import get_model_class
from utils import add_kwargs_to_config

"""
Sparsification: making a transformer model to its Mixture-of-Sparsities version.
1) Reorder the heads and neurons for efficient sparsity indexing.
2) Add sparsity map to config for module-wise sparsity.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Sparsifying a transformers model.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of pretrained model, for indexing model class.",   
    )
    parser.add_argument( # We'd better download the model for ease of use.
        "--teacher_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",    
    )
    parser.add_argument(
        "--record_path_or_regex",
        type=str,
        required=True,
        help="Where to load the records.",
    )
    parser.add_argument( # NIL for distillation.
        "--data_type",
        type=str,
        required=True,
        help="Type of formatted data, for indexing data pipeline.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./", 
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the evaluation loader.",
    )
    parser.add_argument("--use_act_ckpt", action="store_true", help="Use Activation Checkpointing or not.")
    parser.add_argument("--use_bf16", action="store_true", help="Use BF16 or not.")
    parser.add_argument("--deepspeed", type=str, required=True, help="Path to deepspeed config.")
    parser.add_argument("--model_suffix", type=str, default="none", help="Suffix for outputs.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])

    args.output_dir = os.path.join(args.output_dir, f"{args.model_type}-{args.model_suffix}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize DDP.
    deepspeed.init_distributed(dist_backend="nccl")
    # Pin GPU to be used to process local rank (one GPU per process).
    torch.cuda.set_device(args.local_rank)
    is_main = (args.local_rank == -1 or dist.get_rank() == 0)
    device = torch.device("cuda")

    ds_config = json.load(open(args.deepspeed, "r"))
    ds_config.update(**{"train_micro_batch_size_per_gpu": args.per_device_eval_batch_size, "gradient_accumulation_steps": 1, "gradient_clipping": 1})
    if args.use_bf16:
        ds_config.pop("fp16")
        ds_config.update(**{"bf16": {"enabled": True}})
    hfds = HfDeepSpeedConfig(ds_config) # Keep ds_config alive via a weak ref.

    # Load record reader.
    data_reader = TFRecordReader(args.record_path_or_regex, 
        description={"indices": "int"})

    # Get classes which shall be used.
    tokenizer_class, config_class, model_class = get_model_class(args.model_type)
    pipeline_class = get_pipeline_class(args.data_type)

    # MoSification.
    # Load pretrained tokenizer with necessary resizing.
    tokenizer = tokenizer_class.from_pretrained(args.teacher_model_name_or_path, use_fast=not args.use_slow_tokenizer)
    
    # Data pipeline.
    data_pipeline = pipeline_class(tokenizer, args.max_length)

    dev_dataset = TFRecordDistributedDataset(data_reader, shuffle=False)
    dev_loader = DataLoader(dev_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_pipeline.collate)

    config = config_class.from_pretrained(args.teacher_model_name_or_path)
    add_kwargs_to_config(config, use_cache=False)
    model = model_class.from_pretrained(
        args.teacher_model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
    )
    if args.use_act_ckpt:
        # Enable gradient checkpointing for memory efficiency.
        # For backward compatibility.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # HACK: DeepSpeed does not support backward without an effective (resp. dummy) optimizer,
    # so we create an optimizer with zero learning rate here.
    # Hope the zero learning rate would not mess anything : /
    optimizer = FusedAdam(model_parameters, lr=0.0, betas=(0.9, 0.95), eps=1e-8) # Default to AdamW version.
    kwargs = {
        "model": model,
        "model_parameters": model_parameters,
        "config_params": ds_config,
        "optimizer": optimizer,
    }
    ds_engine, _, _, _ = deepspeed.initialize(**kwargs)

    # Sparify!
    print("***** Running sparsification (w. sanity check) *****")
    # Set student to mixture-of-sparsities student with dev set.
    num_layers, num_heads, num_neurons, num_hiddens = \
        config.num_hidden_layers, getattr(config, "num_key_value_heads", config.num_attention_heads), config.intermediate_size, config.hidden_size
    head_importance = torch.zeros(num_layers, num_heads).to(device)
    head_mask = torch.ones(num_layers, num_heads, dtype=torch.bfloat16 if args.use_bf16 else torch.float16).to(device)
    head_mask.requires_grad_(True)
    neuron_importance = torch.zeros(num_layers, num_neurons).to(device)
    neuron_mask = torch.ones(num_layers, num_neurons, dtype=torch.bfloat16 if args.use_bf16 else torch.float16).to(device)
    neuron_mask.requires_grad_(True)
    hidden_importance = torch.zeros(num_hiddens).to(device) # For all
    hidden_mask = torch.ones(num_hiddens, dtype=torch.bfloat16 if args.use_bf16 else torch.float16).to(device)
    hidden_mask.requires_grad_(True)

    input_map = {"text_indices": "input_ids", "text_mask": "attention_mask", "labels": "labels", "label_mask": "label_mask", "lm_labels": "labels"}

    # Compute importance.
    ds_engine.train()
    for batch in dev_loader:
        # batch = [v.to(device) for k, v in batch._asdict().items()]
        batch = {input_map[k]: v.to(device) for k, v in batch._asdict().items()}
        output = ds_engine(**batch, head_mask=head_mask, neuron_mask=neuron_mask, hidden_mask=hidden_mask)
        # loss = F.cross_entropy(output.logits, output.labels, reduction="mean")
        loss = output[0]
        assert torch.isnan(loss) == False, "Loss is NaN!"
        ds_engine.backward(loss)
        head_importance += head_mask.grad.abs().detach() / len(dev_loader)
        neuron_importance += neuron_mask.grad.abs().detach() / len(dev_loader)
        hidden_importance += hidden_mask.grad.abs().detach() / len(dev_loader)
        # Clear the gradients in case of potential overflow.
        head_mask.grad = None
        neuron_mask.grad = None
        hidden_mask.grad = None
        # DeepSpeed's `engine.step()` performs the following operations:
        # - gradient accumulation check
        # - gradient clipping
        # - optimizer step
        # - zero grad
        # - checking overflow
        # - lr_scheduler step (only if engine.lr_scheduler is not None)
        ds_engine.step()
    ds_engine.eval()

    dist.all_reduce(head_importance, op=dist.ReduceOp.SUM)
    head_importance = head_importance / dist.get_world_size()
    dist.all_reduce(neuron_importance, op=dist.ReduceOp.SUM)
    neuron_importance = neuron_importance / dist.get_world_size()
    dist.all_reduce(hidden_importance, op=dist.ReduceOp.SUM)
    hidden_importance = hidden_importance / dist.get_world_size()

    norm_per_layer = torch.pow(torch.pow(head_importance, 2).sum(-1), 0.5)
    head_importance /= norm_per_layer.unsqueeze(-1) + 1e-17
    norm_per_layer = torch.pow(torch.pow(neuron_importance, 2).sum(-1), 0.5)
    neuron_importance /= norm_per_layer.unsqueeze(-1) + 1e-17
    norm_per_layer = torch.pow(torch.pow(hidden_importance, 2).sum(-1), 0.5)
    hidden_importance /= norm_per_layer.unsqueeze(-1) + 1e-17

    # Reorder for efficient indexing with module-wise sparsity.
    del hfds # Disable DeepSpeed environment.
    if is_main:
        # Reload model on main process.
        model = model_class.from_pretrained(
            args.teacher_model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        )
        model = model.to(device)

        base_model = getattr(model, model.base_model_prefix, model)
        head_importance, head_indices = torch.sort(head_importance, dim=1, descending=True)
        neuron_importance, neuron_indices = torch.sort(neuron_importance, dim=1, descending=True)
        hidden_importance, hidden_indices = torch.sort(hidden_importance, dim=0, descending=True)
        head_indices = {layer_idx: indices for layer_idx, indices in enumerate(head_indices)}
        neuron_indices = {layer_idx: indices for layer_idx, indices in enumerate(neuron_indices)}
        hidden_indices = {layer_idx - 1: hidden_indices for layer_idx in range(num_layers + 1)}
        base_model.reorder(head_indices, neuron_indices, hidden_indices)

        # Compute module-wise sparsity from overall sparsity.
        head_sort = [
            (layer_idx, head_importance[layer_idx, head_idx].item())
            for layer_idx in range(num_layers)
            for head_idx in range(num_heads)
        ]
        head_sort = sorted(head_sort, key=lambda x: x[1])
        neuron_sort = [
            (layer_idx, neuron_importance[layer_idx, neuron_idx].item())
            for layer_idx in range(num_layers)
            for neuron_idx in range(num_neurons)
        ]
        neuron_sort = sorted(neuron_sort, key=lambda x: x[1])

        num_total_heads = num_layers * num_heads
        num_total_neurons = num_layers * num_neurons
        sparsity_map = {str(s): {"hidden": {}, "head": {}, "neuron": {}} for s in range(0, 100, 10)}
        # Additional sparsities.
        sparsity_map[str(85)] = {"hidden": {}, "head": {}, "neuron": {}}
        sparsity_map[str(95)] = {"hidden": {}, "head": {}, "neuron": {}}
        sparsity_map[str(96)] = {"hidden": {}, "head": {}, "neuron": {}}
        sparsity_map[str(97)] = {"hidden": {}, "head": {}, "neuron": {}}
        sparsity_map[str(98)] = {"hidden": {}, "head": {}, "neuron": {}}
        sparsity_map[str(99)] = {"hidden": {}, "head": {}, "neuron": {}}
        for sparsity in sparsity_map:
            sqrt_sparsity = 100 - round(100 * math.sqrt(1. - float(sparsity) / 100))
            if sqrt_sparsity > 90:
                head_size = int(config.hidden_size / num_heads)
                heads_sparsified = head_sort[:round(90. / 100 * num_total_heads)]
                additional_num_neurons = round((float(sqrt_sparsity) - 90.) / 100 * num_total_heads * head_size * 4 / 3)
                neurons_sparsified = neuron_sort[:round(float(sqrt_sparsity) / 100 * num_total_neurons) + additional_num_neurons]
            else:
                heads_sparsified = head_sort[:round(float(sqrt_sparsity) / 100 * num_total_heads)]
                neurons_sparsified = neuron_sort[:round(float(sqrt_sparsity) / 100 * num_total_neurons)]
            for (layer_idx, _) in heads_sparsified:
                if str(layer_idx) not in sparsity_map[sparsity]["head"]:
                    sparsity_map[sparsity]["head"][str(layer_idx)] = 0
                sparsity_map[sparsity]["head"][str(layer_idx)] += 1
            for (layer_idx, _) in neurons_sparsified:
                if str(layer_idx) not in sparsity_map[sparsity]["neuron"]:
                    sparsity_map[sparsity]["neuron"][str(layer_idx)] = 0
                sparsity_map[sparsity]["neuron"][str(layer_idx)] += 1
            for layer_idx in range(num_layers + 1):
                sparsity_map[sparsity]["hidden"][str(layer_idx - 1)] = round(float(sqrt_sparsity) / 100 * num_hiddens)

        print("***** Finalizing sparsification *****")
        print("***** Adding sparsity & sparsity map to config *****")
        config.sparsity = "0"
        config.sparsity_map = sparsity_map
        print("***** Saving sparsified model *****")
        save_path = args.output_dir
        tokenizer.save_pretrained(save_path)
        config.save_pretrained(save_path)
        model.save_pretrained(save_path)
    
    # Other processed wait until all finished.
    dist.barrier()
    

if __name__ == "__main__":
    main()
