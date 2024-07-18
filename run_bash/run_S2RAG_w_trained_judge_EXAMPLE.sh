export CUDA_VISIBLE_DEVICES=0

python scripts/S2RAG_inf.py \
--model_name model/Llama-2-7b-chat-hf \
--base Llama2-7B \
--tau 0.45 \
--bertmodel model/en_core_web_sm-3.6.0/en_core_web_sm/en_core_web_sm-3.6.0 \
--trained_path weights_minqi/llama3_8B_256 \
--trained_idx_path data_training/train_data_fusion.jsonl \
--use_trained \
--test \
--choice 0 