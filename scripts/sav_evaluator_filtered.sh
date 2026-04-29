#!/bin/bash
# Evaluation on video_filtered predictions (v1, last-frame backward anchor)
# GT: video_filtered/Annotations
# Hard/Easy split: each video is divided into 3 temporal segments;
# the segment with the lowest mean IoU is "hard", the other two are "easy".

GT_ROOT=/root/workspace/d663ovsp420c73cg8l00/data/CVPR2029_aug8/test/video/Annotations
EVAL="python sav_dataset/sav_evaluator_hard_easy.py --strict --do_not_skip_first_and_last_frame"
PRED_ROOT=/root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video_filtered

# -----------------------------------------------------------------------
# 非 fusion 模式
# -----------------------------------------------------------------------

# echo "========== 0216 (non-fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0216_v1

# echo "========== 0217 (non-fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0217_v1

# echo "========== 0218 (non-fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0218_v1

# echo "========== 0220_1 boundary_weight (non-fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0220_1_v1

# echo "========== 0220_2 small_area_penalty (non-fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0220_2_v1

# # -----------------------------------------------------------------------
# # fusion 模式
# # -----------------------------------------------------------------------

# echo "========== 0216 (fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0216_v1_fusion

# echo "========== 0217 (fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0217_v1_fusion

# echo "========== 0218 (fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0218_v1_fusion

# echo "========== 0220_1 boundary_weight (fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0220_1_v1_fusion

# echo "========== 0220_2 small_area_penalty (fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0220_2_v1_fusion

# echo "========== 0220_2 划分数据集 12 8 (fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0222_12_8_decoder

# echo "========== 0220_2 划分数据集 8 12 decider (fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0222_8_12_fusion

# echo "========== 0220_2 划分数据集 8 12 decider (fusion) =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/img
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/fwd
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/bwd
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/fused
# echo "========== 0220_2 划分数据集 8 12 baseline =========="
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/img
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/fwd
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/bwd
# $EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/fused
# echo "========== 0225 划分数据集 8 12 fusion 版本 =========="
$EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0226_zeroshot

$EVAL --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}/Predictions_0226_8_12_new_fusion_penalty








