export WANDB_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
export WANDB_MODE="offline"
export SWANLAB_API_KEY="xHqvDmJAYUBxDJPr4vwbz"
export SWANLAB_MODE="local"
export ENTITY="yunyang"
# export PROJECT='test'
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TASK_QUEUE_ENABLE=0
export HCCL_OP_BASE_FFTS_MODE_ENABLE=0
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export HCCL_ALGO="level0:NA;level1:H-D_R"
# --machine_rank=${MACHINE_RANK} \ 
# --main_process_ip=${MAIN_PROCESS_IP_VALUE} \ 

export MASTER_HOST="$VC_WORKER_HOSTS"
export MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
export NNODES="$MA_NUM_HOSTS"
export NODE_RANK="$VC_TASK_INDEX"
# also indicates NPU per node
export NGPUS_PER_NODE="$MA_NUM_GPUS"
export NUM_PROCESSES=$(($NGPUS_PER_NODE * $NNODES))
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=7200

export MASTER_PORT="${MASTER_PORT}"
export PROJECT="${PROJECT_NAME}"

export OUTPUT_DIR="/home/ma-user/work/checkpoint/gyy/runs/$PROJECT"

# set npu plog env
ma_vj_name=`echo ${MA_VJ_NAME} | sed 's:ma-job:modelarts-job:g'`
task_name="worker-${VC_TASK_INDEX}"
task_plog_path=${MA_LOG_DIR}/${ma_vj_name}/${task_name}

mkdir -p ${task_plog_path}
export ASCEND_PROCESS_LOG_PATH=${task_plog_path}

echo "plog path: ${ASCEND_PROCESS_LOG_PATH}"


accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_ma.yaml \
    --machine_rank=${NODE_RANK} \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    --num_machines=${NNODES} \
    --num_processes=${NUM_PROCESSES} \
    opensora/train/train_t2v_diffusers_ema_lb.py \
    --ema_deepspeed_config_file scripts/accelerate_configs/zero3_npu.json \
    --model OpenSoraT2V_v1_5-6B/122 \
    --text_encoder_name_1 google/t5-v1_1-xl \
    --cache_dir "../../cache_dir/" \
    --text_encoder_name_2 laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
    --cache_dir "../../cache_dir/" \
    --dataset t2v \
    --data scripts/train_data/current_hq_on_npu9.txt \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/home/ma-user/work/checkpoint/pretrained/vae_32dim" \
    --sample_rate 1 \
    --num_frames 1 \
    --max_hxw 65536 \
    --min_hxw 36864 \
    --force_5_ratio \
    --gradient_checkpointing \
    --train_batch_size=16 \
    --dataloader_num_workers 16 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant_with_warmup" \
    --mixed_precision="bf16" \
    --report_to="swanlab" \
    --checkpointing_steps=500 \
    --allow_tf32 \
    --model_max_length 512 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --resume_from_checkpoint="latest" \
    --drop_short_ratio 1.0 \
    --hw_stride 16 \
    --train_fps 16 \
    --seed 1234 \
    --group_data \
    --use_decord \
    --output_dir=$OUTPUT_DIR \
    --vae_fp32 \
    --rf_scheduler \
    --proj_name "$PROJECT" \
    --log_name "$PROJECT" \
    --skip_abnormal_step --ema_decay_grad_clipping 0.99 \
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
