export PROJECT=$PROJECT_NAME
WEIGHT_PATH="/home/opensora/pre_weights/"
env
export WANDB_MODE='offline'
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export HCCL_ALGO="level0:NA;level1:H-D_R"

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example_by_deepspeed.yaml \
    --machine_rank=${MACHINE_RANK} \
    --main_process_ip=${MAIN_PROCESS_IP_VALUE} \
    opensora/train/train_t2v_diffusers.py \
    --model OpenSoraT2V-ROPE-L/122 \
    --text_encoder_name ${WEIGHT_PATH}/google/mt5-xxl \
    --cache_dir "../cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "${WEIGHT_PATH}/test140k/" \
    --video_data "./scripts/train_data/video_data_on_npu.txt" \
    --image_data "./scripts/train_data/image_data_on_npu.txt" \
    --sample_rate 1 \
    --num_frames 1 \
    --max_height 480 \
    --max_width 640 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=8 \
    --dataloader_num_workers 20 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=4e-5 \
    --lr_scheduler="cosine" \
    --seed=10 \
    --lr_warmup_steps=500 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=2000 \
    --output_dir="/home/image_data/checkpoints/${PROJECT}/" \
    --allow_tf32 \
    --model_max_length 512 \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --use_rope \
    --noise_offset 0.02 \
    --pretrained "/home/image_data/checkpoints/image3d_rope_256p_zp_umt5/checkpoint-146000/model_ema/diffusion_pytorch_model.safetensors" \
    --resume_from_checkpoint="latest"