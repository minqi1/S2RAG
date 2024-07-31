export CUDA_VISIBLE_DEVICES=0

python scripts/S2RAG_inf.py \
--model_name model/Llama-2-7b-chat-hf \
--tau 0.45 \
--bertmodel model/en_core_web_sm \
--choice 0 