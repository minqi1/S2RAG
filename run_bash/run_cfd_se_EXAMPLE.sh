export CUDA_VISIBLE_DEVICES=0
# for simplicity, task is modified to choice: 0: pqa, 1: tqa, 2: pubhealth, 3: arc
python /home/minqi/code/S2RAG/scripts/cfd_se_collection.py \
--model_name /home/minqi/code/S2RAG/model/Llama-2-7b-chat-hf \
--choice 0 
