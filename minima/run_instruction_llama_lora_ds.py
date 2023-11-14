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
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

import deepspeed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.ops.adam import FusedAdam

import transformers
from transformers import AdamW, Adafactor, get_scheduler
from transformers import BitsAndBytesConfig
from transformers.deepspeed import HfDeepSpeedConfig

from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_int8_training, get_peft_model_state_dict

from tqdm.auto import tqdm
from filelock import FileLock

from data import get_pipeline_class, TFRecordReader, TFRecordDataset, TFRecordDistributedDataset
from models import get_model_class
from utils import set_seed, add_kwargs_to_config, keep_recent_ckpt, find_most_recent_ckpt, Logger, AverageMeter

from torch.utils.tensorboard import SummaryWriter


logger = Logger()


def gather(tensor, num_examples=None):
    output_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # output = concat[:num_examples] # Truncate dummy elements added by DistributedSampler.
    output = concat
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on an instruction-following task with LoRA.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of pretrained model, for indexing model class.",   
    )
    parser.add_argument( # We'd better download the model for ease of use.
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",    
    )
    parser.add_argument(
        "--train_record_path_or_regex",
        type=str,
        required=True,
        help="Where to load the train records.",
    )
    parser.add_argument(
        "--dev_record_path_or_regex",
        type=str,
        required=True,
        help="Where to load the dev records.",
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
    parser.add_argument("--num_patience_epochs", type=int, default=3, help="Total number of patience epochs for early stop.")
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
    parser.add_argument("--use_int8", action="store_true", help="Use INT8 or not.")
    parser.add_argument("--use_int4", action="store_true", help="Use INT8 or not.")
    parser.add_argument("--resume", action="store_true", help="Resume training or not.")
    parser.add_argument("--use_bf16", action="store_true", help="Use BF16 or not.")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU or not.")
    parser.add_argument("--deepspeed", type=str, required=True, help="Path to deepspeed config.")
    parser.add_argument("--model_suffix", type=str, default="none", help="Suffix for outputs.")
    
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
    is_int8 = args.use_int8
    is_int4 = args.use_int4
    # assert not is_int8, "INT8 is not compatible with DeepSpeed?"
    device = torch.device("cpu") if args.use_cpu else torch.device("cuda")

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
        summary = SummaryWriter(os.path.join(os.environ["QS_LOG_DIR"], os.environ["TRIAL_NAME"]))
    else:
        transformers.utils.logging.set_verbosity_error()
        logger.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    logger.info("***** Configuration ready! *****")

    # Load record reader.
    train_data_reader = TFRecordReader(args.train_record_path_or_regex, 
        description={"indices": "int", "mask": "int"})
    dev_data_reader = TFRecordReader(args.dev_record_path_or_regex, 
        description={"indices": "int", "mask": "int"})
    
    logger.info("***** Data ready! *****")

    # Get classes which shall be used.
    tokenizer_class, config_class, model_class = get_model_class(args.model_type)
    pipeline_class = get_pipeline_class(args.data_type)

    # Load pretrained tokenizer with necessary resizing.
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    
    # Data pipeline.
    data_pipeline = pipeline_class(tokenizer, args.max_length)

    train_dataset = TFRecordDistributedDataset(train_data_reader, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, collate_fn=data_pipeline.collate)
    
    dev_dataset = TFRecordDistributedDataset(dev_data_reader, shuffle=False)
    dev_loader = DataLoader(dev_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_pipeline.collate)
       
    # Initialize, then rewrite or add kwargs of original config for distillation alignment.
    config = config_class.from_pretrained(args.model_name_or_path)
    add_kwargs_to_config(config, use_cache=False)
    if is_int8:
        int8_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head",]) # Skip a few modules to make them trainable.
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            quantization_config=int8_config,
        )
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True) # True or False?
    elif is_int4:
        int4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", llm_int8_skip_modules=["embed_tokens", "lm_head"]) # Skip a few modules to make them trainable.
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            quantization_config=int4_config,
        )
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True) # True or False?    
    else:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        )
    model.resize_token_embeddings(len(tokenizer))
    # Configurate LoRA following Baize.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj",),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=("lm_head",), # For lm_head, `modules_to_save` will be marked as trainable.
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm", "ln", "layer_norm", "layernorm", "norm"]
    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    # optimizer = AdamW(grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    optimizer = FusedAdam(grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8) # Default to AdamW version.

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.num_grad_accum_steps)
    num_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_patience_steps = args.num_patience_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    kwargs = {
        "model": model,
        "model_parameters": model_parameters,
        "config_params": ds_config,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    ds_engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)

    if args.resume:
        args.model_name_or_path = find_most_recent_ckpt(args.output_dir)
        # This magically updates optimizer and lr_scheduler.
        load_path, _ = ds_engine.load_checkpoint(
            args.model_name_or_path, load_optimizer_states=True, load_lr_scheduler_states=True)
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
    best_dev_step = 0
    best_dev_metric = {}

    num_saved_steps = 0
    if args.resume:
        num_saved_steps = int(os.path.basename(args.resume).split("-")[1])

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_loader):
            if num_completed_steps < num_saved_steps:
                if step % args.num_grad_accum_steps == 0 or step == len(train_loader) - 1:
                    progress_bar.update(1)
                    num_completed_steps += 1
                continue

            ds_engine.train()
            batch = [v.to(device) for k, v in batch._asdict().items()]
            beginner.record()
            output = model.Output(**ds_engine(batch))
            loss = output[0]
            loss_ = loss.detach().clone()
            dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
            train_losses.update(loss_.item() / dist.get_world_size())
            loss = loss / args.num_grad_accum_steps
            ds_engine.backward(loss)
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
            ds_engine.step()
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
                    model.eval()
                    with torch.no_grad():
                        losses = []
                        for batch in dev_loader:
                            batch = [v.to(device) for k, v in batch._asdict().items()]
                            output = model.Output(**ds_engine(batch))
                            loss = model.loss_fn(output)
                            losses.append(loss.item())

                    perplexity = torch.tensor(np.exp2(np.mean(losses)), device=device)
                    dist.all_reduce(perplexity, op=dist.ReduceOp.SUM)
                    perplexity = perplexity.item() / dist.get_world_size()
                    dev_metric = {"perplexity": perplexity, "neg_perplexity": -perplexity}
                    logger.info(f"  Train loss = {train_losses.avg}")
                    logger.info(f"  Train throughput = {train_throughputs.avg}")
                    logger.info(f"  Dev metric = {dev_metric}")

                    if not best_dev_metric or dev_metric[args.selection_metric] > best_dev_metric[args.selection_metric]:
                        logger.info("***** Saving best *****")
                        best_dev_step = num_completed_steps
                        best_dev_metric.update(**dev_metric)
                        
                        time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                        save_dir = os.path.join(args.output_dir, \
                            f"ckpt-{num_completed_steps}-{time_stamp}")
                        os.makedirs(save_dir, exist_ok=True)
                        ds_engine.save_checkpoint(save_dir)
                        # Save LoRA. Weights might be a placeholder in zero3 and need a gather.
                        if ds_engine.zero_optimization_partition_weights():
                            state_dict = ds_engine._zero3_consolidated_16bit_state_dict()
                        else:
                            state_dict = ds_engine.module.state_dict()
                        state_dict = get_peft_model_state_dict(model, state_dict)
                        if is_main:
                            torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
                        if is_main:
                            lora_config.save_pretrained(save_dir)
                            tokenizer.save_pretrained(save_dir)
                            config.save_pretrained(save_dir)
                            keep_recent_ckpt(args.output_dir, 5)

            if num_completed_steps - best_dev_step >= num_patience_steps:
                logger.info("***** Early stopping *****")
                break
        # If early stop, then break the outer loop.
        else:
            continue
        break  

    logger.info("***** Finalizing training *****")
    logger.info(f"  Best dev step = {best_dev_step}")
    logger.info(f"  Best dev metric = {best_dev_metric}")
    

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