python scripts/S2RAG_fullprocess_demo.py \
--model_name model/Meta-Llama-3-8B-Instruct \
--tau 0.45 \
--bertmodel model/en_core_web_sm \
--input_q_ctx data_eval/demo_q_ctx_fantasy.jsonl \
--use_trained \
--demo_task fantasy \
--use_default_prompt

