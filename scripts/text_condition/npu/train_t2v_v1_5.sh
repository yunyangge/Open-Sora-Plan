export WANDB_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
export WANDB_MODE="online"
export ENTITY="yunyang"
# export PROJECT=$PROJECT_NAME
export PROJECT='test'
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
# --deepspeed_zero2_config.yaml

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config_npu.yaml \
    opensora/train/train_t2v_diffusers.py \
    --model OpenSoraT2V_v1_5-5B/122 \
    --text_encoder_name_1 google/t5-v1_1-xl \
    --cache_dir "../../cache_dir/" \
    --text_encoder_name_2 laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
    --cache_dir "../../cache_dir/" \
    --dataset t2v \
    --data "scripts/train_data/video_data_debug_on_npu.txt" \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/home/save_dir/lzj/formal_888" \
    --vae_fp32 \
    --sample_rate 1 \
    --num_frames 105 \
    --force_resolution \
    --max_height 768 \
    --max_width 768 \
    --snr_gamma 5.0 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --allow_tf32 \
    --model_max_length 512 \
    --cfg 0.1 \
    --speed_factor 1.0 \
    --drop_short_ratio 0.0 \
    --hw_stride 16 \
    --sparse1d \
    --train_fps 16 \
    --seed 1234 \
    --trained_data_global_step 0 \
    --group_data \
    --use_decord \
    --prediction_type "v_prediction" \
    --v1_5_scheduler \
    --output_dir="/home/save_dir/runs/$PROJECT" \
    # --resume_from_checkpoint="latest" \
    # --force_resolution