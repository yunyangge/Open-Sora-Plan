export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29512 \
    -m opensora.sample.sample \
    --model_path /home/save_dir/runs/t2v_1_5_dit_bs16x32x8_lr1e-4_256x256_192x192_new6b/checkpoint-6000/model \
    --version v1_5 \
    --num_frames 1 \
    --height 256 \
    --width 256 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/home/save_dir/pretrained/t5/t5-v1_1-xl" \
    --text_encoder_name_2 "/home/save_dir/pretrained/clip/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189" \
    --text_prompt examples/sora.txt \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/home/save_dir/lzj/Middle888" \
    --save_img_path "./t2v_new_script_6k" \
    --fps 18 \
    --guidance_scale 7.0 \
    --num_sampling_steps 28 \
    --max_sequence_length 512 \
    --sample_method FlowMatchEulerDiscrete \
    --seed 1234 \
    --num_samples_per_prompt 1 \