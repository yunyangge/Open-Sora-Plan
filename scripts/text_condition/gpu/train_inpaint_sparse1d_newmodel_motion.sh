export WANDB_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
export WANDB_MODE="online"
export ENTITY="yunyang"
export PROJECT="inpaint_93x1280x1280_stage3_gpu"
# export PROJECT="test"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export PDSH_RCMD_TYPE=ssh
# NCCL setting
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=25
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_RETRY_CNT=32
# export NCCL_ALGO=Tree

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example.yaml \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "../../cache_dir/" \
    --dataset inpaint \
    --data "scripts/train_data/video_data_high_aes_720p.txt" \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/Causal-Video-VAE/results/WFVAE_DISTILL_FORMAL" \
    --sample_rate 1 \
    --num_frames 93 \
    --max_height 1280 \
    --max_width 1280 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=100 \
    --allow_tf32 \
    --model_max_length 512 \
    --use_image_num 0 \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --noise_offset 0.0 \
    --use_rope \
    --skip_low_resolution \
    --speed_factor 1.0 \
    --ema_decay 0.9999 \
    --drop_short_ratio 0.0 \
    --hw_stride 32 \
    --sparse1d --sparse_n 4 \
    --use_motion \
    --train_fps 16 \
    --seed 1234 \
    --trained_data_global_step 0 \
    --group_data \
    --use_decord \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr \
    --t2v_ratio 0.05 \
    --i2v_ratio 0.5 \
    --transition_ratio 0.3 \
    --v2v_ratio 0.05 \
    --clear_video_ratio 0.0 \
    --min_clear_ratio 0 \
    --default_text_ratio 0.5 \
    --pretrained_transformer_model_path "/storage/gyy/hw/Open-Sora-Plan/runs/inpaint_93x640x640_stage3_gpu/checkpoint-14777/model_ema" \
    --output_dir runs/$PROJECT 2>&1 | tee training_log_new.txt
    # --resume_from_checkpoint="latest" \
