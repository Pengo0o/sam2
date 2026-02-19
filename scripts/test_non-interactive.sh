#正常非交互式模式
python tools/vos_inference_non_interactive.py \                                                                                            
    --sam2_cfg configs/sam2.1_training/sam2.1_hiera_b+_non_interactive_fusion.yaml \                                                         
    --sam2_checkpoint checkpoints/my_lora_checkpoint.pt \
    --base_video_dir /data/DAVIS/JPEGImages/480p \
    --output_mask_dir /tmp/results_standard \
    --score_thresh 0.0 \
    --lora_rank 8 \
    --lora_dropout 0.1

# fusion模式
python tools/vos_inference_non_interactive.py \
    --sam2_cfg configs/sam2.1_training/sam2.1_hiera_b+_non_interactive_fusion.yaml \
    --sam2_checkpoint checkpoints/my_lora_checkpoint.pt \
    --base_video_dir /data/DAVIS/JPEGImages/480p \
    --output_mask_dir /tmp/results_fusion \
    --use_fusion \
    --score_thresh 0.0 \
    --lora_rank 8 \
    --lora_dropout 0.1

# 指定视频列表文件，并启用每个对象的PNG文件输出和后处理步骤
python tools/vos_inference_non_interactive.py \
    --sam2_cfg configs/sam2.1_training/sam2.1_hiera_b+_non_interactive_fusion.yaml \
    --sam2_checkpoint checkpoints/my_lora_checkpoint.pt \
    --base_video_dir /data/DAVIS/JPEGImages/480p \
    --output_mask_dir /tmp/results_fusion \
    --video_list_file val_videos.txt \
    --use_fusion \
    --per_obj_png_file \
    --apply_postprocessing