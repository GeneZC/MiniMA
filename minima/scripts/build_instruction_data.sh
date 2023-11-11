
python run_building_data_minichat.py \
    --input_dir dir/to/sharegpt \
    --input_regex "*.jsonl" \
    --output_dir dir/to/builded/sharegpt \
    --tokenizer_name_or_path path/to/minima-3b \
    --do_lower_case \
    --max_seq_length 4096 \
    --num_processors 16