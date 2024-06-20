# export WANDB_KEY=""
# export ENTITY="linbin"
# export PROJECT="65x512x512_10node_bs2_lr2e-5_4img"
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint_XL_122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "/data01/transition/cache_dir" \
    --dataset inpaint \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/data01/transition/Open-Sora-Plan_models/vae" \
    --video_data "scripts/train_data/video_data.txt" \
    --sample_rate 1 \
    --num_frames 65 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=4 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=100000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=1 \
    --output_dir="inpaint_65x512x512_1node_bs2_lr2e-5" \
    --allow_tf32 \
    --use_deepspeed \
    --model_max_length 300 \
    --enable_tiling \
    --pretrained "/data01/transition/Open-Sora-Plan_models/65x512x512/diffusion_pytorch_model.safetensors" \
    --validation_dir "/data01/transition/Open-Sora-Plan/validition_dir" \
    --i2v_ratio 0.4 \
    --transition_ratio 0.4 \
    --text_drop_rate 0.1 \
    --default_text_ratio 0.1 \
    
