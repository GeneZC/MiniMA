# Towards the Law of Capacity Gap in Distilling Language Models

<img src="assets/logo.jpg" alt="logo" width="400"/>

📑 [arXiv]() | 🤗 [HuggingFace]() ｜ 🤖 [ModelScope]()

Language model (LM) distillation is a trending area that aims to distil the knowledge resided in a large teacher LM to a small student one. While various methods have been proposed to push the distillation to its limits, it is still a pain distilling LMs when a large capacity gap is exhibited between the teacher and the student LMs. The pain is mainly resulted by the curse of capacity gap, which describes that a larger teacher LM cannot always lead to a better student LM than one distilled from a smaller teacher LM due to the affect of capacity gap increment. That is, there is likely an optimal point yielding the best student LM along the scaling course of the teacher LM. Even worse, the curse of capacity gap can be only partly yet not fully lifted as indicated in previous studies.

However, the tale is not ever one-sided. Although a larger teacher LM has better performance than a smaller teacher LM, it is much more resource-demanding especially in the context of recent large LMs (LLMs). Consequently, instead of sticking to lifting the curse, leaving the curse as is should be arguably fine. Even better, in this paper, we reveal that the optimal capacity gap is almost consistent across different student scales and architectures, fortunately turning the curse into the law of capacity gap. The law later guides us to distil a 3B student LM (termed MiniMA) from a 7B teacher LM (adapted LLaMA2-7B). MiniMA is demonstrated to yield a new compute-performance pareto frontier among existing 3B LMs on commonly used benchmarks, and its instruction-tuned version (termed MiniChat) outperforms a wide range of 3B competitors in GPT4 evaluation and could even compete with several 7B chat models. 

<img src="assests/teaser_a.jpg" alt="teaser_a" width="400" /> <img src="assests/teaser_b.jpg" alt="teaser_b" width="400" />

## 🔗 Quick Links

- [Updates](#updates)
- [Quick Start](#quick-start)
- [Tutorials](#tutorials)
  - [Requirements](#requirements)
  - [Adaptation](#adaptation)
  - [Distillation](#distillation)
  - [Finetuning](#finetuning)
  - [Evaluation](#evaluation)
- [Future Work](#future-work)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Updates

[2023/11] We have released the paper, updated the codebase, uploaded the checkpoints.

## Quick Start

Since MiniMA and MiniChat follow architecture of LLaMA, they can be seamlessly used in HuggingFace Transformers with minimal requirements.

The following is an example code snippet to use MiniMA and MiniChat:

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

# MiniMA
tokenizer = AutoTokenizer.from_pretrained("GeneZC/MiniMA-3B", use_fast=False)
# GPU.
model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniMA-3B", use_cache=True, device_map="auto", torch_dtype=torch.float16).eval()
# CPU.
# model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniMA-3B", use_cache=True, device_map="cpu", torch_dtype=torch.float16).eval()

prompt = "Question: Sherrie tells the truth. Vernell says Sherrie tells the truth. Alexis says Vernell lies. Michaela says Alexis tells the truth. Elanor says Michaela tells the truth. Does Elanor tell the truth?\nAnswer: No\n\nQuestion: Kristian lies. Sherrie says Kristian lies. Delbert says Sherrie lies. Jerry says Delbert tells the truth. Shalonda says Jerry tells the truth. Does Shalonda tell the truth?\nAnswer: No\n\nQuestion: Vina tells the truth. Helene says Vina lies. Kandi says Helene tells the truth. Jamey says Kandi lies. Ka says Jamey lies. Does Ka tell the truth?\nAnswer: No\n\nQuestion: Christie tells the truth. Ka says Christie tells the truth. Delbert says Ka lies. Leda says Delbert tells the truth. Lorine says Leda tells the truth. Does Lorine tell the truth?\nAnswer:"
input_ids = tokenizer([prompt]).input_ids
output_ids = model.generate(
    torch.as_tensor(input_ids).cuda(),
    do_sample=True,
    temperature=0.7,
    max_new_tokens=1024,
)
output_ids = output_ids[0][len(input_ids[0]):]
output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
# output: "No"

from conversation import get_default_conv_template

# MiniChat
tokenizer = AutoTokenizer.from_pretrained("GeneZC/MiniChat-3B", use_fast=False)
# GPU.
model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-3B", use_cache=True, device_map="auto", torch_dtype=torch.float16).eval()
# CPU.
# model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-3B", use_cache=True, device_map="cpu", torch_dtype=torch.float16).eval()

conv = get_default_conv_template("minichat")

question = "Implement a program to find the common elements in two arrays without using any extra data structures."
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer([prompt]).input_ids
output_ids = model.generate(
    torch.as_tensor(input_ids).cuda(),
    do_sample=True,
    temperature=0.7,
    max_new_tokens=1024,
)
output_ids = output_ids[0][len(input_ids[0]):]
output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
# output: "def common_elements(arr1, arr2):\n    if len(arr1) == 0:\n        return []\n    if len(arr2) == 0:\n        return arr1\n\n    common_elements = []\n    for element in arr1:\n        if element in arr2:\n            common_elements.append(element)\n\n    return common_elements"
# Multiturn conversation could be realized by continuously appending questions to `conv`.
```

## Tutorials

The reproductions of MiniMA and MiniChat follow several stages including adaptation of LLaMA2, distillation to MiniMA, and finetuning to MiniChat.

### Requirements

### Adaptation

In case of potential Chinese applications, we adapt LLaMA2 to Chinese without degrading English performance.

**Tokenizer**

The following is an example script (i.e., scripts/llama_expand_vocab.sh) to expand vocabulary:
```bash
python run_expanding_vocab_llama.py \
    --tokenizer_dir path/to/llama2-7b \
    --chinese_sp_path path/to/chinese_cp 
```

**Data**

The following is an example script (i.e., scripts/llama_build_data.sh) to build adaptation data (e.g., Pile):
```bash
python run_building_data_llama.py \
    --input_dir "dir/to/pile" \
    --input_regex "*.jsonl" \
    --output_dir dir/to/builded/pile \
    --tokenizer_name_or_path path/to/minima \
    --do_lower_case \
    --max_seq_length 4096 \
    --num_processors 32
```

**Training**

The following is an example script (i.e., scripts/llama_lm_7b_ds.sh) to adapt LLaMA2 with 32 Nvdia A100 GPUs:
```bash
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$GPU_NUM --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pretraining_llama_ds.py \
    --model_type llama_lm \
    --model_name_or_path /mnt/nlp-ali/usr/zhangchen3/plms/llama2-7b-ada-init \
    --record_path_or_regex "dir/to/builded/pile/*.tfrecord" \
    --data_type llama_lm \
    --output_dir path/to/save/outputs \
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

### Distillation

**Data**

The distillation reuses the adaptation data.

**Training**

The following is an example script (i.e., scripts/llama_logit_3b_from_7b_ds.sh) to distil from LLaMA2-7B to MiniMA with 16 Nvdia A100 GPUs:
```bash
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$GPU_NUM --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_distillation_llama_ds.py \
    --model_type llama_lm \
    --teacher_model_name_or_path path/to/llama2-7b-ada \
    --student_model_name_or_path path/to/minima-3b-init \
    --record_path_or_regex "/mnt/nlp-ali/usr/zhangchen3/builded_*_llama_4096_minichat/*.tfrecord" \
    --data_type llama_lm \
    --output_dir /mnt/nlp-ali/usr/zhangchen3/qs_outputs \
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

### Finetuning

**Data**

The following is an example script (i.e., scripts/llama_build_data_minichat.sh) to build adaptation data (e.g., Pile):
```bash
python run_building_data_minichat.py \
    --input_dir minichat \
    --input_regex "*.jsonl" \
    --output_dir  \
    --tokenizer_name_or_path 
    --do_lower_case \
    --max_seq_length 4096 \
    --num_processors 16
```

**Training**

```bash
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$GPU_NUM --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_instruction_llama_ds.py \
    --model_type llama_instruct \
    --model_name_or_path x \
    --train_record_path_or_regex x \
    --dev_record_path_or_regex x \
    --data_type llama_instruct \
    --output_dir utputs \
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
    --model_suffix 3b
```

### Evaluation

We mainly refer to the following repositories for evaluation:
- [MMLU](https://github.com/hendrycks/test)
- [CEval](https://github.com/hkust-nlp/ceval)
- [DROP,BBH,HumanEval](https://github.com/declare-lab/instruct-eval)
- [GSM8K](https://github.com/Guangxuan-Xiao/GSM8K-eval)

## Future Work

- More diverse blend of data sources, e.g., Chinese wikipedia, books, etc.
- Smaller models, e.g., 1.2B model, say MicoMA.
- Preference optimization, e.g., DPO to MiniChat.

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Chen (chenzhang9702@outlook.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!


## Citation

Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{zhang2023law,
    title={Towards the Law of Capacity Gap in Distilling Language Models},
    author={Zhang, Chen and Song, Dawei and Ye, Zheyu and Gao, Yan},
    year={2023},
    url={}
}
```

