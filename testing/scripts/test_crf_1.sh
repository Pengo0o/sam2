# 1129
CUDA_VISIBLE_DEVICES=1 python testing/test_sam2_lora_with_visual_crf.py \
    --checkpoint output_1129_sam2.1_hiera_l_hels_finetune+lora+new_loss_with_mask+dis5k+boundary_weight/checkpoints/checkpoint_300.pt \
    --output-dir testing/output_1129_CRF_3_sam2.1_hiera_l_hels_finetune+lora+new_loss_with_mask+dis5k+boundary_weight \
    --visualize

# 1205
CUDA_VISIBLE_DEVICES=1 python testing/test_sam2_lora_with_visual_crf.py \
    --checkpoint output_1205_sam2.1_hiera_l_hels_finetune+lora+new_loss_with_mask+dis5k+boundary_weight/checkpoints/checkpoint_300.pt \
    --output-dir testing/output_1205_CRF_3_sam2.1_hiera_l_hels_finetune+lora+new_loss_with_mask+dis5k+boundary_weight \
    --visualize

# 1213
CUDA_VISIBLE_DEVICES=1 python testing/test_sam2_lora_with_visual_crf.py \
    --checkpoint output_1213_sam2.1_hiera_l_hels_finetune+lora_+new_loss_+boundary_weight+small_area_penalty/checkpoints/checkpoint_300.pt \
    --output-dir testing/output_1213_CRF_3_sam2.1_hiera_l_hels_finetune+lora_+new_loss_+boundary_weight+small_area_penalty \
    --visualize \