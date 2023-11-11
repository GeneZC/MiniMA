
python run_building_data_llama.py \
    --input_dir "dir/to/wudao" \
    --input_regex "*.jsonl" \
    --output_dir dir/to/builded/wudao \
    --tokenizer_name_or_path path/to/llama2-7b-ada \
    --do_lower_case \
    --max_seq_length 4096 \
    --num_processors 32