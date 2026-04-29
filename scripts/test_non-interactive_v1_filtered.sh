#!/bin/bash
# Non-interactive VOS inference v1 (last-frame backward anchor)
# Data: video_filtered
# -----------------------------------------------------------------------

BASE_VIDEO_DIR=/root/workspace/d663ovsp420c73cg8l00/data/CVPR2029_aug8/test/video/JPEGImages
CKPT_ROOT=/root/workspace/d663ovsp420c73cg8l00/code/sam2
OUT_ROOT=/root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video_filtered

# -----------------------------------------------------------------------
# 非 fusion 模式
# -----------------------------------------------------------------------

# 0216
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0216_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+small_area_penalty_non_interactive/checkpoints/checkpoint_200.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0216_v1 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# # 0217
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0217_sam2.1_hiera_l_hels_finetune+VOS+lora_+baseline/checkpoints/checkpoint_200.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0217_v1 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# # 0218
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0218_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+small_area_penalty_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0218_v1 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# # 0220 ablation 1 (boundary_weight)
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0220_1_v1 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# # 0220 ablation 2 (small_area_penalty)
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+small_area_penalty_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0220_2_v1 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# # -----------------------------------------------------------------------
# # fusion 模式
# # -----------------------------------------------------------------------

# # 0216 fusion
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0216_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+small_area_penalty_non_interactive/checkpoints/checkpoint_200.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0216_v1_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --use_fusion \
#     --per_obj_png_file

# # 0217 fusion
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0217_sam2.1_hiera_l_hels_finetune+VOS+lora_+baseline/checkpoints/checkpoint_200.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0217_v1_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --use_fusion \
#     --per_obj_png_file

# # 0218 fusion
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0218_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+small_area_penalty_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0218_v1_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --use_fusion \
#     --per_obj_png_file

# # 0220 ablation 1 (boundary_weight) fusion
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0220_1_v1_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --use_fusion \
#     --per_obj_png_file

# # 0220 ablation 2 (small_area_penalty) fusion
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0220_ablation_sam2.1_hiera_l_hels_finetune+VOS+lora_+small_area_penalty_non_interactive_fusion/checkpoints/checkpoint_200.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0220_2_v1_fusion \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --use_fusion \
#     --per_obj_png_file

# 8 12划分结果
# python tools/vos_inference_non_interactive_v1.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0222_sam2.1_hiera_l_hels_finetune+VOS+lora_+baseline_8_12/checkpoints/checkpoint_120.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0222_8_12 \
#     --score_thresh 0.0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --per_obj_png_file

# 8 12 划分 冻住更多 module
# python tools/vos_inference_non_interactive_v1_no_lora.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0224_sam2.1_hiera_l_hels_finetune+VOS+new_baseline_freeze_more_modules_8_12/checkpoints/checkpoint_120.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0222_12_8_decoder \
#     --score_thresh 0.0 \
#     --per_obj_png_file

# 8 12 划分使用 fusion
# python tools/vos_inference_non_interactive_v1_no_lora.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0224_final_sam2.1_hiera_l_hels_finetune+VOS+freeze_more_modules_8_12+_non_interactive_fusion/checkpoints/checkpoint_120.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0222_8_12_fusion \
#     --score_thresh 0.0 \
#     --per_obj_png_file
# 8 12 划分 使用 fusion 以及 small area penalty

# python tools/vos_inference_non_interactive_v1_no_lora.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0224_final_sam2.1_hiera_l_hels_finetune+VOS+freeze_more_modules_8_12+_non_interactive_fusion_small_area_penalty/checkpoints/checkpoint_120.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0222_8_12_fusion_small_area_penalty \
#     --score_thresh 0.0 \
#     --per_obj_png_file

# python tools/vos_inference_non_interactive_v2_no_lora.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint ${CKPT_ROOT}/output_0225_final_sam2.1_hiera_l_hels_finetune+VOS+freeze_more_modules_8_12+_non_interactive_new_fusion_detach_video/checkpoints/checkpoint_120.pt \
#     --base_video_dir ${BASE_VIDEO_DIR} \
#     --output_mask_dir ${OUT_ROOT}/Predictions_0225_8_12_new_fusion \
#     --score_thresh 0.0 \
#     --per_obj_png_file \
#     --use_fusion

python tools/vos_inference_non_interactive_v2_no_lora.py \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam2_checkpoint ${CKPT_ROOT}/checkpoints/sam2.1_hiera_large.pt \
    --base_video_dir ${BASE_VIDEO_DIR} \
    --output_mask_dir ${OUT_ROOT}/Predictions_0226_zeroshot \
    --score_thresh 0.0 \
    --per_obj_png_file
    
    

python tools/vos_inference_non_interactive_v2_no_lora.py \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam2_checkpoint ${CKPT_ROOT}/output_0226_final_sam2.1_hiera_l_hels_finetune+VOS+freeze_more_modules_8_12+_non_interactive_new_fusion_detach_video_small_area_penalty/checkpoints/checkpoint_120.pt \
    --base_video_dir ${BASE_VIDEO_DIR} \
    --output_mask_dir ${OUT_ROOT}/Predictions_0226_8_12_new_fusion_penalty \
    --score_thresh 0.0 \
    --per_obj_png_file \
    --use_fusion