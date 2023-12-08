
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$GPU_NUM --nnodes=$NODE_WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_sparsification_llama.py \
    --model_type sparsellama_lm \
    --teacher_model_name_or_path path/to/llama2-7b-ada \
    --record_path_or_regex "dir/to/builded/part-of-wudao/*.tfrecord" \
    --data_type llama_lm \
    --output_dir dir/to/outputs \
    --max_length 512 \
    --per_device_eval_batch_size 8 \
    --use_act_ckpt \
    --use_bf16 \
    --deepspeed ds_config.json \
    --model_suffix 7b