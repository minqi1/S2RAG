export CUDA_VISIBLE_DEVICES=0
python scripts/explanation_collection.py \
--model_name model/Llama-2-7b-chat-hf \
--input_file data_training/train_data_fusion_example.jsonl \
--output_file data_training/train_data_fusion_w_exp_example.json \
--batch_size 4 \
