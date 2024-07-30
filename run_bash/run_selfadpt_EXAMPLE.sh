export CUDA_VISIBLE_DEVICES=0

python scripts/selfadpt_collection.py \
--model_name model/Llama-2-7b-chat-hf \
--input_file data_eval/popqa_longtail_w_gs_example.jsonl \
--output_file minqi_inf_output/llama2chat-pqa-selfadp.json \
--max_new_tokens 20