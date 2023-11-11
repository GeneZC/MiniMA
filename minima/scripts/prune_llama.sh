
python run_sparsification_llama.py \
    --model_type sparsellama_lm \
    --teacher_model_name_or_path path/to/llama2-7b-ada \
    --record_path_or_regex "dir/to/builded/wudao/*.tfrecord" \
    --data_type llama_lm \
    --output_dir dir/to/outputs \
    --max_length 512 \
    --per_device_eval_batch_size 2 \
    --model_suffix 7b