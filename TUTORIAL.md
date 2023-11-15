## Tutorials

> [!NOTE]  
> The code mainly lies in `minima`.
> The content is under construction. Read with cautions.

### Outline

- [Requirements](#requirements)
- [Adaptation](#adaptation)
- [Distillation](#distillation)
- [Finetuning](#finetuning)
- [Evaluation](#evaluation)

### Requirements

Basically, requirements are emphasized for TFRecord IO, HuggingFace Transformers, Flash Attention, and DeepSpeed.

You can directly install the required packages with the following command:

```bash
sh scripts/install_requirements.sh
```

### Adaptation

In case of potential Chinese applications, LLaMA is adapted to Chinese without English performance degraded.

**Tokenizer**

The vocabulary should be expanded via manipulating LLaMA's tokenizer.

The following is an example script (i.e., `scripts/expand_vocab.sh`) to expand vocabulary:

```bash
python run_expanding_vocab_llama.py \
    --llama_tokenizer_dir path/to/llama2-7b \
    --chinese_sp_path path/to/chinese_sp.model
```

Replace the original `tokenizer.model` in `llama2-7b` with the new `tokenizer.model`, then you get `llama2-7b-ada-init`

**Data**

The adaptation data should be builded into TFRecords for efficient IO.

The following is an example script (i.e., `scripts/build_pretraining_data.sh`) to build adaptation data (e.g., WuDao):

```bash
python run_building_data_llama.py \
    --input_dir "dir/to/wudao" \
    --input_regex "*.jsonl" \
    --output_dir dir/to/builded/wudao \
    --tokenizer_name_or_path path/to/llama2-7b-ada-init \
    --do_lower_case \
    --max_seq_length 4096 \
    --num_processors 32
```

**Training**

The training is executed with 32 Nvdia A100 GPUs.

The following is an example script (i.e., `scripts/adapt_llama.sh`) to adapt `llama2-7b-ada-init`:

```bash
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$GPU_NUM --nnodes=$NODE_WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pretraining_llama_ds.py \
    --model_type llama_lm \
    --model_name_or_path llama2-7b-ada-init \
    --record_path_or_regex "dir/to/builded/wudao/*.tfrecord" \
    --data_type llama_lm \
    --output_dir dir/to/outputs \
    --max_length 4096 \
    --per_device_train_batch_size 4 \
    --num_grad_accum_steps 8 \
    --per_device_eval_batch_size 128 \
    --learning_rate 3e-5 \
    --weight_decay 1e-1 \
    --log_interval 100 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --warmup_proportion 0.01 \
    --max_grad_norm 1.0 \
    --seed 776 \
    --resume \
    --use_act_ckpt \
    --use_bf16 \
    --deepspeed ds_config.json \
    --model_suffix 7b_ada
```

Now you get `llama2-7b-ada`.

### Distillation

**Pruning**

The pruning is executed with 1 Nvidia A100 GPU and only a small portion of adaptation data.

The following is an example script (i.e., `scripts/prune_llama.sh`) to prune `llama2-7b-ada`:

```bash
python run_sparsification_llama.py \
    --model_type sparsellama_lm \
    --teacher_model_name_or_path path/to/llama2-7b-ada \
    --record_path_or_regex "dir/to/builded/wudao/*.tfrecord" \
    --data_type llama_lm \
    --output_dir dir/to/outputs \
    --max_length 512 \
    --per_device_eval_batch_size 2 \
    --model_suffix 7b
```

Convert the sparse checkpoint to a 3B one, then you get `minima-3b-init`.

The following is an example script (i.e., `scripts/convert_llama.sh`) to obtain `minima-3b-init`:

```bash
python run_converting_llama.py
    --sparsellama_dir sparsellama2-7b-ada
```



**Data**

The distillation reuses the adaptation data.

**Training**

The training is executed with 16 Nvdia A100 GPUs.

The following is an example script (i.e., `scripts/distil_minima.sh`) to distil from `llama2-7b-ada` to `minima-3b`:

```bash
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$GPU_NUM --nnodes=$NODE_WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_distillation_llama_ds.py \
    --model_type llama_lm \
    --teacher_model_name_or_path path/to/llama2-7b-ada \
    --student_model_name_or_path path/to/minima-3b-init \
    --record_path_or_regex "dir/to/builded/wudao/*.tfrecord" \
    --data_type llama_lm \
    --output_dir dir/to/outputs \
    --max_length 4096 \
    --per_device_train_batch_size 8 \
    --num_grad_accum_steps 8 \
    --per_device_eval_batch_size 128 \
    --learning_rate 3e-4 \
    --weight_decay 1e-1 \
    --log_interval 100 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --warmup_proportion 0.01 \
    --max_grad_norm 1.0 \
    --seed 776 \
    --resume \
    --use_act_ckpt \
    --use_bf16 \
    --deepspeed ds_config.json \
    --layer 24 \
    --hidden 3072 \
    --head 24 \
    --intermediate 8192 \
    --model_suffix 3b_from_7b
```

Now you get `minima-3b`.

### Finetuning

**Data**

The instruction data should also be builded into TFRecords for efficient IO. And the instruction data should be formatted in the following style:

```json
[ # list of conversations
    [ # list of turns in a conversation
        { # utterances in a turn
            "User": "Create an array of length 5 which contains all even numbers between 1 and 10.",
            "Assistant": "arr = [2, 4, 6, 8, 10]"
        }
    ],
    ...
]
```

The following is an example script (i.e., `scripts/build_instruction_data.sh`) to build instruction data (e.g., ShareGPT):

```bash
python run_building_data_minichat.py \
    --input_dir dir/to/sharegpt \
    --input_regex "*.jsonl" \
    --output_dir dir/to/builded/sharegpt \
    --tokenizer_name_or_path path/to/minima-3b \
    --do_lower_case \
    --max_seq_length 4096 \
    --num_processors 16
```

**Training**

The training is executed with 8 Nvdia A100 GPUs.

The following is an example script (i.e., `scripts/instrcut_minichat.sh`) to finetune `minima-3b` to `minichat-3b`:

```bash
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$GPU_NUM --nnodes=$NODE_WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_instruction_llama_ds.py \
    --model_type llama_instruct \
    --model_name_or_path path/to/minima-3b \
    --train_record_path_or_regex "dir/to/sharegpt/*.train.tfrecord" \
    --dev_record_path_or_regex "dir/to/sharegpt/*.dev.tfrecord" \
    --data_type llama_instruct \
    --output_dir dir/to/outputs \
    --max_length 4096 \
    --per_device_train_batch_size 16 \
    --num_grad_accum_steps 2 \
    --per_device_eval_batch_size 64 \
    --learning_rate 2e-5 \
    --weight_decay 1e-1 \
    --log_interval 100 \
    --num_train_epochs 3 \
    --num_patience_epochs 2 \
    --lr_scheduler_type cosine \
    --warmup_proportion 0.1 \
    --max_grad_norm 1.0 \
    --seed 776 \
    --resume \
    --use_act_ckpt \
    --use_bf16 \
    --deepspeed ds_config.json \
    --model_suffix 3b_minichat
```

And the untested script (i.e., `scripts/instrcut_minichat_lora.sh`) to finetune `minima-3b` to `minichat-3b` with LoRA:
```bash
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$GPU_NUM --nnodes=$NODE_WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_instruction_llama_lora_ds.py \
    --model_type llama_instruct \
    --model_name_or_path path/to/minima-3b \
    --train_record_path_or_regex "dir/to/sharegpt/*.train.tfrecord" \
    --dev_record_path_or_regex "dir/to/sharegpt/*.dev.tfrecord" \
    --data_type llama_instruct \
    --output_dir dir/to/outputs \
    --max_length 4096 \
    --per_device_train_batch_size 16 \
    --num_grad_accum_steps 2 \
    --per_device_eval_batch_size 64 \
    --learning_rate 2e-5 \
    --weight_decay 1e-1 \
    --log_interval 100 \
    --num_train_epochs 3 \
    --num_patience_epochs 2 \
    --lr_scheduler_type cosine \
    --warmup_proportion 0.1 \
    --max_grad_norm 1.0 \
    --seed 776 \
    --resume \
    --use_bf16 \
    --deepspeed ds_config.json \
    --model_suffix 3b_minichat_lora
```

### Evaluation

We mainly refer to the following repositories for evaluation:
- MMLU: https://github.com/hendrycks/test
- CEval: https://github.com/hkust-nlp/ceval
- DROP, BBH, HumanEval: https://github.com/declare-lab/instruct-eval
- GSM8K: https://github.com/Guangxuan-Xiao/GSM8K-eval
- Vicuna-Bench, BELLE-Bench: https://github.com/Neutralzz/BiLLa/tree/main/eval_codes