#正常非交互式模式
# python tools/vos_inference_non_interactive.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# # fusion模式
# python tools/vos_inference_non_interactive.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions \
#     --use_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file \
#     --video_list_file /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/val_video.txt

# # 指定视频列表文件，并启用每个对象的PNG文件输出和后处理步骤
# python tools/vos_inference_non_interactive.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint checkpoints/my_lora_checkpoint.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video \
#     # --video_list_file val_videos.txt \
#     --use_fusion \
#     --per_obj_png_file \
#     --apply_postprocessing

# python tools/vos_inference_non_interactive_debug.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Debug \
#     --use_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file \
#     --video_list_file /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/val_video.txt

# python tools/vos_inference_non_interactive_debug.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Debug \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --video_list_file /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/val_video.txt


#+-----------------------------------------------------------------------
# 非fusion
# python tools/vos_inference_non_interactive_v2.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0216_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+small_area_penalty_non_interactive/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0216 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# python tools/vos_inference_non_interactive_v2.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0217_sam2.1_hiera_l_hels_finetune+VOS+lora_+baseline/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0217 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# python tools/vos_inference_non_interactive_v2.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0218_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+small_area_penalty_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0218 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# python tools/vos_inference_non_interactive_v2.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0220_1 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# python tools/vos_inference_non_interactive_v2.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+small_area_penalty_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0220_2 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file


# # fusion模式
# python tools/vos_inference_non_interactive_v2.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0218_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+small_area_penalty_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0218_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --use_fusion \
#     --per_obj_png_file

# python tools/vos_inference_non_interactive_v2.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0220_1_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --use_fusion \
#     --per_obj_png_file

# python tools/vos_inference_non_interactive_v2.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+small_area_penalty_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0220_2_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --use_fusion \
#     --per_obj_png_file

# python tools/vos_inference_non_interactive_v2.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0216_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+small_area_penalty_non_interactive/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0216_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --use_fusion \
#     --per_obj_png_file

# python tools/vos_inference_non_interactive_v2.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0217_sam2.1_hiera_l_hels_finetune+VOS+lora_+baseline/checkpoints/checkpoint_200.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0217_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --use_fusion \
#     --per_obj_png_file


# python tools/vos_inference_non_interactive_no_lora_debug.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0224_final_sam2.1_hiera_l_hels_finetune+VOS+freeze_more_modules_8_12+_non_interactive_fusion/checkpoints/checkpoint_120.pt \
#     --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029_aug8/test/video/JPEGImages \
#     --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029_aug8/test/video/Debug \
#     --score_thresh 0.0 
    # --per_obj_png_file



python tools/vos_inference_non_interactive_no_lora_debug.py \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam2_checkpoint /root/workspace/d663ovsp420c73cg8l00/code/sam2/output_0224_sam2.1_hiera_l_hels_finetune+VOS+new_baseline_freeze_more_modules_8_12/checkpoints/checkpoint_120.pt \
    --base_video_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029_aug8/test/video/JPEGImages \
    --output_mask_dir /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029_aug8/test/video/Debug_baseline \
    --score_thresh 0.0 
