
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
    --model_suffix 3b_from_7b_minima

# Params
# - Embedding 32000 * 3072 = 98,304,000
# - Layer 24 * 113246208 = ~2.7e9
#   - Self-Attention 4 * 3072 * 3072 = 37,748,736
#   - MLP 3 * 3072 * 8192 = 75,497,472
#   - Sum 37748736 + 75497472 = 113,246,208
# - Output 98,304,000
# - Sum ~2.7e9 + ~0.2e9 = 2.9e9 = 2.9B