export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=8B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
LORA_RANK=256
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file scripts/stage3_no_offloading_accelerate.conf \
    scripts/finetune_rmst_lla3.py \
    --model_name_or_path model/Meta-Llama-3-8B-Instruct \
    --tokenizer_name model/Meta-Llama-3-8B-Instruct \
    --use_slow_tokenizer \
    --use_left_padding \
    --train_file data_training/train_data_fusion_w_exp.jsonl \
    --max_seq_length 1024  \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1.5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir weights_minqi/llama3_${MODEL_SIZE}_${LORA_RANK}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --checkpointing_steps 600 \
    --use_lora \
    --lora_rank $LORA_RANK 

