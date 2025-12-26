#!/bin/bash

# SAM2 + CRF 批量测试脚本

cd /opt/data/private/hyp/sam2

# 1112
python testing/test_sam2_lora_with_visual_crf.py \
    --checkpoint output_1112_sam2.1_hiera_l_hels_finetune+lora/checkpoints/checkpoint_300.pt \
    --output-dir testing/output_1112_CRF_3_sam2.1_hiera_l_hels_finetune+lora \
    --visualize

# 1119
python testing/test_sam2_lora_with_visual_crf.py \
    --checkpoint output_1119_sam2.1_hiera_l_hels_finetune+lora+new_loss_with_mask/checkpoints/checkpoint_300.pt \
    --output-dir testing/output_1119_CRF_3_sam2.1_hiera_l_hels_finetune+lora+new_loss_with_mask \
    --visualize

# 1121
python testing/test_sam2_lora_with_visual_crf.py \
    --checkpoint output_1121_sam2.1_hiera_l_hels_finetune+lora+new_loss_with_mask+dis5k/checkpoints/checkpoint_300.pt \
    --output-dir testing/output_1121_CRF_3_sam2.1_hiera_l_hels_finetune+lora+new_loss_with_mask+dis5k \
    --visualize



echo "所有实验完成!"
