export CUDA_VISIBLE_DEVICES=0

python /home/minqi/code/S2RAG/scripts/S2RAG_inf.py \
--model_name /home/minqi/code/S2RAG/model/Llama-2-7b-chat-hf \
--tau 0.5 \
--bertmodel /home/minqi/code/S2RAG/model/en_core_web_sm-3.6.0/en_core_web_sm/en_core_web_sm-3.6.0 \
--choice 3 