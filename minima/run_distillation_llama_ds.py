# -*- coding: utf-8 -*-

import os
import time
import glob
import math
import json
import functools
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

import deepspeed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

import transformers
from transformers import AdamW, Adafactor, get_scheduler
from transformers import BitsAndBytesConfig
from transformers.deepspeed import HfDeepSpeedConfig

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from filelock import FileLock

from data import get_pipeline_class, TFRecordReader, TFRecordDataset, TFRecordDistributedDataset
from models import get_model_class
from utils import set_seed, add_kwargs_to_config, keep_recent_ckpt, find_most_recent_ckpt, Logger, AverageMeter

from torch.utils.tensorboard import SummaryWriter

logger = Logger()


input_map = {"text_indices": "input_ids", "text_mask": "attention_mask", "labels": "labels", "label_mask": "label_mask", "lm_labels": "labels"}


def acc(preds, labels):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    return {"acc": acc}


def gather(tensor, num_examples=None):
    output_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # output = concat[:num_examples] # Truncate dummy elements added by DistributedSampler.
    output = concat
    return output


def soft_cross_entropy(input, target, reduction="mean"):
    s_likelihood = F.log_softmax(input, dim=-1)
    t_probability = F.softmax(target, dim=-1)
    cross_entropy = -torch.sum(t_probability * s_likelihood, dim=-1)
    if reduction == "mean":
        cross_entropy = cross_entropy.mean()
    else:
        pass
    return cross_entropy


def parse_args():
    parser = argparse.ArgumentParser(description="Distilling a transformers model.")
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
        "--student_model_name_or_path",
        type=str,
        required=True,
        help="Path to configurated model.",   
    )
    parser.add_argument(
        "--record_path_or_regex",
        type=str,
        required=True,
        help="Where to load the records.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        help="Type of formatted data, for indexing data pipeline.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs", 
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
        "--per_device_train_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the training loader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the evaluation loader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--log_interval", type=int, default=1000, help="Interval of logging and possible saving.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--num_grad_accum_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_proportion", type=float, default=0.01, help="Proportion of the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum norm of gradients."
    )
    parser.add_argument("--seed", type=int, default=776, help="A seed for reproducible training.")
    # parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--resume", action="store_true", help="Resume training or not.")
    parser.add_argument("--use_act_ckpt", action="store_true", help="Use Activation Checkpointing or not.")
    parser.add_argument("--use_bf16", action="store_true", help="Use BF16 or not.")
    parser.add_argument("--use_offload", action="store_true", help="Use CPU Offload or not.")
    parser.add_argument("--deepspeed", type=str, required=True, help="Path to deepspeed config.")
    parser.add_argument("--model_suffix", type=str, default="none", help="Suffix for outputs.")

    parser.add_argument("--sparsity", type=int, default=0, help="Sparsity.")
    parser.add_argument("--layer", type=int, default=4, help="Layer.")
    parser.add_argument("--hidden", type=int, default=384, help="Hidden.")
    parser.add_argument("--head", type=int, default=12, help="Head.")
    parser.add_argument("--intermediate", type=int, default=1536, help="Intermediate.")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])

    args.output_dir = os.path.join(args.output_dir, f"{args.model_type}_{args.model_suffix}")
    os.makedirs(args.output_dir, exist_ok=True)
    args.resume = args.resume and find_most_recent_ckpt(args.output_dir) is not None

    # Initialize DeepSpeed.
    deepspeed.init_distributed(dist_backend="nccl")
    # Pin GPU to be used to process local rank (one GPU per process).
    torch.cuda.set_device(args.local_rank) 
    is_main = (args.local_rank == -1 or dist.get_rank() == 0)
    device = torch.device("cuda") #if torch.cuda.is_available() else torch.device("cpu")

    ds_config = json.load(open(args.deepspeed, "r"))
    ds_config.update(**{"train_micro_batch_size_per_gpu": args.per_device_train_batch_size, "gradient_accumulation_steps": args.num_grad_accum_steps, "gradient_clipping": args.max_grad_norm})
    if args.use_bf16:
        ds_config.pop("fp16")
        ds_config.update(**{"bf16": {"enabled": True}})
    hfds = HfDeepSpeedConfig(ds_config) # Keep ds_config alive via a weak ref.

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.add_stream_handler()
    logger.add_file_handler(args.output_dir)
    if is_main:
        transformers.utils.logging.set_verbosity_warning()
        logger.set_verbosity_info()
        summary = SummaryWriter(os.path.join(os.environ["LOG_DIR"], os.environ["TRIAL_NAME"]))
    else:
        transformers.utils.logging.set_verbosity_error()
        logger.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    logger.info("***** Configuration ready! *****")

    # Load record reader.
    if "flan" in args.model_type:
        data_reader = TFRecordReader(args.record_path_or_regex, 
            description={"indices": "int", "mask": "int"})
    else:
        data_reader = TFRecordReader(args.record_path_or_regex, 
            description={"indices": "int"})
    
    logger.info("***** Data ready! *****")

    # Get classes which shall be used.
    tokenizer_class, config_class, model_class = get_model_class(args.model_type)
    pipeline_class = get_pipeline_class(args.data_type)

    # Load pretrained tokenizer with necessary resizing.
    tokenizer = tokenizer_class.from_pretrained(args.teacher_model_name_or_path, use_fast=not args.use_slow_tokenizer)
    
    # Data pipeline.
    data_pipeline = pipeline_class(tokenizer, args.max_length)

    train_dataset = TFRecordDistributedDataset(data_reader, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, collate_fn=data_pipeline.collate)
    
    # dev_dataset = TFRecordDistributedDataset(dev_reader, shuffle=False)
    # dev_loader = DataLoader(dev_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_pipeline.collate)
        
    # Initialize, then rewrite or add kwargs of original config for distillation alignment.
    t_config = config_class.from_pretrained(args.teacher_model_name_or_path)
    add_kwargs_to_config(t_config, use_cache=False)
    t_model = model_class.from_pretrained(
        args.teacher_model_name_or_path,
        config=t_config,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
    )
    s_config = config_class.from_pretrained(args.student_model_name_or_path)
    add_kwargs_to_config(s_config, use_cache=False)
    if args.sparsity != 0:
        add_kwargs_to_config(s_config, sparsity=str(args.sparsity))
    else:
        add_kwargs_to_config(s_config, num_hidden_layers=args.layer)
        add_kwargs_to_config(s_config, hidden_size=args.hidden)
        add_kwargs_to_config(s_config, num_attention_heads=args.head)
        add_kwargs_to_config(s_config, intermediate_size=args.intermediate)
    s_model = model_class.from_pretrained(
        args.student_model_name_or_path,
        config=s_config,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
    )   
    s_model.resize_token_embeddings(len(tokenizer))
    if args.use_act_ckpt:
        # Enable gradient checkpointing for memory efficiency.
        # For backward compatibility.
        if hasattr(s_model, "enable_input_require_grads"):
            s_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            s_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        s_model.gradient_checkpointing_enable()

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm", "ln", "layer_norm", "layernorm", "norm"]
    grouped_parameters = [
        {
            "params": [p for n, p in s_model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in s_model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    # optimizer = AdamW(grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    if args.use_offload:
        optimizer = DeepSpeedCPUAdam(grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8) # Default to AdamW version.
    else:
        optimizer = FusedAdam(grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8) # Default to AdamW version.

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.num_grad_accum_steps)
    num_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    t_model_parameters = list(filter(lambda p: p.requires_grad, t_model.parameters()))
    kwargs = {
        "model": t_model,
        "model_parameters": t_model_parameters,
        "config_params": ds_config,
    }
    t_ds_engine, _, _, _ = deepspeed.initialize(**kwargs)

    s_model_parameters = list(filter(lambda p: p.requires_grad, s_model.parameters()))
    kwargs = {
        "model": s_model,
        "model_parameters": s_model_parameters,
        "config_params": ds_config,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    s_ds_engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)

    if args.resume:
        args.student_model_name_or_path = find_most_recent_ckpt(args.output_dir)
        # This magically updates optimizer and lr_scheduler.
        load_path, _ = s_ds_engine.load_checkpoint(
            args.student_model_name_or_path, load_optimizer_states=True, load_lr_scheduler_states=True)
        assert load_path is not None, f"Failed to resume from checkpoint {load_path}."

    logger.info("***** Model & Opitimizer & Scaler ready! *****")

    
    # Train!
    total_batch_size = args.per_device_train_batch_size * args.num_grad_accum_steps
    total_batch_size = total_batch_size * dist.get_world_size()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation, parallel & distributed) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.num_grad_accum_steps}")
    logger.info(f"  Total optimization steps = {num_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(num_train_steps), disable=not is_main)
    num_completed_steps = 0
    train_losses = AverageMeter(args.num_grad_accum_steps)
    beginner, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    train_throughputs = AverageMeter(args.num_grad_accum_steps)

    num_saved_steps = 0
    if args.resume:
        num_saved_steps = int(os.path.basename(args.student_model_name_or_path).split("-")[1])

    t_ds_engine.eval()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_loader):
            if num_completed_steps < num_saved_steps:
                if step % args.num_grad_accum_steps == 0 or step == len(train_loader) - 1:
                    progress_bar.update(1)
                    num_completed_steps += 1
                continue

            s_ds_engine.train()
            batch = {input_map[k]: v.to(device) for k, v in batch._asdict().items()}
            beginner.record()
            with torch.no_grad():
                t_output = t_ds_engine(**batch)
            s_output = s_ds_engine(**batch)
            loss = 0.5 * s_output[0] + 0.5 * soft_cross_entropy(s_output[1], t_output[1].detach())
            loss_ = loss.detach().clone()
            dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
            train_losses.update(loss_.item() / dist.get_world_size())
            loss = loss / args.num_grad_accum_steps
            s_ds_engine.backward(loss)
            ender.record()
            torch.cuda.synchronize()
            elapsed_time = beginner.elapsed_time(ender)
            throughput = total_batch_size * args.max_length / args.num_grad_accum_steps / elapsed_time 
            train_throughputs.update(throughput)
            # DeepSpeed's `engine.step()` performs the following operations:
            # - gradient accumulation check
            # - gradient clipping
            # - optimizer step
            # - zero grad
            # - checking overflow
            # - lr_scheduler step (only if engine.lr_scheduler is not None)
            s_ds_engine.step()
            if step % args.num_grad_accum_steps == 0 or step == len(train_loader) - 1:
                # DeepSpeed does itw own stepping, including gradient clipping and zeroing, etc.
                # So we do nothing here.
                progress_bar.update(1)
                num_completed_steps += 1
                if is_main:
                    summary.add_scalar("loss/train", train_losses.avg, num_completed_steps)
            
                if num_completed_steps % args.log_interval == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info(f"  Num completed epochs = {epoch}")
                    logger.info(f"  Num completed steps = {num_completed_steps}")
                    # model.eval()
                    # with torch.no_grad():
                    #     preds, labels = [], []
                    #     for batch in dev_loader:
                    #         batch = [v.to(device) for k, v in batch._asdict().items()]
                    #         output = model(batch)
                    #         pred, label = output.prediction, output.label
                    #         if is_dist:
                    #             preds.extend(gather(pred).cpu().numpy().tolist())
                    #             labels.extend(gather(label).cpu().numpy().tolist())
                    #         else:
                    #             preds.extend(pred.cpu().numpy().tolist())
                    #             labels.extend(label.cpu().numpy().tolist())

                    dev_metric = {}
                    dev_metric.update(**{"loss": train_losses.avg})
                    logger.info(f"  Train loss = {train_losses.avg}")
                    logger.info(f"  Train throughput = {train_throughputs.avg}")
                    # logger.info(f"  Dev metric = {dev_metric}")

                    logger.info("***** Saving the current *****")
                    time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                    save_dir = os.path.join(args.output_dir, \
                        f"ckpt-{num_completed_steps}-{time_stamp}")
                    os.makedirs(save_dir, exist_ok=True)
                    s_ds_engine.save_checkpoint(save_dir)
                    # Save. Weights might be a placeholder in zero3 and need a gather.
                    if s_ds_engine.zero_optimization_partition_weights():
                        state_dict = s_ds_engine._zero3_consolidated_16bit_state_dict()
                    else:
                        state_dict = s_ds_engine.module.state_dict()
                    if is_main:
                        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
                    if is_main:
                        tokenizer.save_pretrained(save_dir)
                        s_config.save_pretrained(save_dir)
                        keep_recent_ckpt(args.output_dir, 10)

    logger.info("***** Finalizing training *****")
    logger.info("***** Saving the last *****")
    # time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    last_dir = os.path.join(args.output_dir, \
        "ckpt-last")
    os.makedirs(last_dir, exist_ok=True)
    s_ds_engine.save_checkpoint(last_dir)
    # Save. Weights might be a placeholder in zero3 and need a gather.
    if s_ds_engine.zero_optimization_partition_weights():
        state_dict = s_ds_engine._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = s_ds_engine.module.state_dict()
    if is_main:
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    if is_main:
        tokenizer.save_pretrained(last_dir)
        s_config.save_pretrained(last_dir)
    

if __name__ == "__main__":
    """
    1. Single-Node multi-process distributed training

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
                arguments of your training script)

    2. Multi-Node multi-process distributed training: (e.g. two nodes)


    Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
                --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
                and all other arguments of your training script)

    Node 2:

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
                --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
                and all other arguments of your training script)
    """
    main()