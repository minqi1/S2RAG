python scripts/S2RAG_fullprocess_demo.py \
--model_name model/Llama-2-7b-chat-hf \
--tau 0.45 \
--bertmodel model/en_core_web_sm-3.6.0/en_core_web_sm/en_core_web_sm-3.6.0 \
--input_q_ctx data_eval/demo_q_ctx.jsonl \
--demo_task fact \
--use_default_prompt

