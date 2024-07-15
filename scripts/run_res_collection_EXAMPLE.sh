export CUDA_VISIBLE_DEVICES=0

python /home/minqi/code/S2RAG/scripts/response_collection.py \
--model_name /home/minqi/code/S2RAG/model/Llama-2-7b-chat-hf \
--input_file /home/minqi/code/S2RAG/data_eval/arc_challenge_processed_example.jsonl \
--output_file /home/minqi/code/S2RAG/scripts/minqi_inf_output/Llama-2-7b-arc.json \
--task arc_c \
--max_new_tokens 20 \
--use_default_prompt


