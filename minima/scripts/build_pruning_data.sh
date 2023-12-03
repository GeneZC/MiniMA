
python run_building_data_llama.py \
    --input_dir "dir/to/part-of-wudao" \
    --input_regex "*.jsonl" \
    --output_dir dir/to/builded/part-of-wudao \
    --tokenizer_name_or_path path/to/llama2-7b-ada-init \
    --do_lower_case \
    --max_seq_length 512 \
    --num_processors 32