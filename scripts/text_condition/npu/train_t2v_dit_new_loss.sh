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
# --deepspeed_zero2_config_npu.yaml

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config_npu.yaml \
    opensora/train/train_t2v_diffusers_ema_new_loss.py \
    --ema_deepspeed_config_file scripts/accelerate_configs/zero3_npu.json \
    --model OpenSoraT2V_DiT_2B/122 \
    --text_encoder_name_1 google/t5-v1_1-xl \
    --cache_dir "../../cache_dir/" \
    --text_encoder_name_2 laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
    --cache_dir "../../cache_dir/" \
    --dataset t2v \
    --data scripts/train_data/current_hq_on_npu_dit_new_loss.txt \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/home/save_dir/lzj/Middle888" \
    --sample_rate 1 \
    --num_frames 1 \
    --max_hxw 65536 \
    --min_hxw 36864 \
    --force_5_ratio \
    --gradient_checkpointing \
    --train_batch_size=32 \
    --dataloader_num_workers 20 \
    --learning_rate=2e-5 \
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
    --output_dir="/home/save_dir/runs/SUV/dit_new_loss/$PROJECT" \
    --vae_fp32 \
    --rf_scheduler \
    --proj_name "$PROJECT" \
    --log_name "$PROJECT" \
    --skip_abnorml_step --ema_decay_grad_clipping 0.99 \
    --trained_data_global_step 0 \
    --use_ema \
    --ema_update_freq 50 \
    --ema_decay 0.99 \
    # --enable_tiling \
    # --resume_from_checkpoint="latest" \
    # --max_hxw 65536 \
    # --min_hxw 36864 \
    # --force_5_ratio \
    # --force_resolution \
    # --max_height 768 \
    # --max_width 768 \
