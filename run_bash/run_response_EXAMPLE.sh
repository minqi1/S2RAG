export CUDA_VISIBLE_DEVICES=0

python /home/minqi/code/S2RAG/scripts/response_collection.py \
--model_name /home/minqi/code/S2RAG/model/Llama-2-7b-chat-hf \
--input_file /home/minqi/code/S2RAG/data_eval/popqa_longtail_w_gs_example.jsonl \
--output_file /home/minqi/code/S2RAG/minqi_inf_output/llama2chat-health-origin.json \
--max_new_tokens 20 \
--use_default_prompt
