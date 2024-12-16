export WANDB_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
export WANDB_MODE="online"
export ENTITY="yunyang"
export PROJECT=$PROJECT_NAME
# export PROJECT='test'
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

export TASK_QUEUE_ENABLE=0
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export HCCL_ALGO="level0:NA;level1:H-D_R"
# --machine_rank=${MACHINE_RANK} \ 
# --main_process_ip=${MAIN_PROCESS_IP_VALUE} \ 
# --multi_node_example_by_deepspeed.yaml

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example_by_deepspeed.yaml \
    --num_machines=${NUM_MACHINES} \
    --num_processes=${NUM_PROCESSES} \
    --machine_rank=${MACHINE_RANK} \
    --main_process_ip=${MAIN_PROCESS_IP_VALUE} \
    opensora/train/train_t2v_diffusers_ema_lb.py \
    --ema_deepspeed_config_file scripts/accelerate_configs/zero3_npu.json \
    --model OpenSoraT2V_SUV_2B/122 \
    --text_encoder_name_1 google/t5-v1_1-xl \
    --cache_dir "../../cache_dir/" \
    --text_encoder_name_2 laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
    --cache_dir "../../cache_dir/" \
    --dataset t2v \
    --data scripts/train_data/current_hq_on_npu_suv.txt \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/home/save_dir/lzj/Middle888" \
    --sample_rate 1 \
    --num_frames 65 \
    --force_resolution \
    --max_height 288 \
    --max_width 512 \
    --train_video_only \
    --max_h_div_w_ratio=0.875 \
    --min_h_div_w_ratio=0.5 \
    --max_hxw 147456 \
    --min_hxw 147456 \
    --gradient_checkpointing \
    --train_batch_size=2 \
    --dataloader_num_workers 10 \
    --learning_rate=1e-4 \
    --adam_weight_decay=1e-4 \
    --adam_epsilon=1e-15 \
    --lr_scheduler="constant_with_warmup" \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=1000 \
    --allow_tf32 \
    --model_max_length 512 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --resume_from_checkpoint="latest" \
    --drop_short_ratio 1.0 \
    --hw_stride 16 \
    --train_fps 16 \
    --seed 1024 \
    --group_data \
    --use_decord \
    --output_dir="/home/save_dir/runs/SUV/suv/$PROJECT" \
    --vae_fp32 \
    --rf_scheduler \
    --proj_name "$PROJECT" \
    --log_name "$PROJECT" \
    --skip_abnormal_step --ema_decay_grad_clipping 0.99 \
    --trained_data_global_step 0 \
    --use_ema \
    --ema_update_freq 1 \
    --ema_decay 0.9999 \
    # --enable_tiling \
    # --force_5_ratio \
    # --resume_from_checkpoint="latest" \
    # --max_hxw 65536 \
    # --min_hxw 36864 \
    # --force_5_ratio \
    # --force_resolution \
    # --max_height 768 \
    # --max_width 768 \
