# 基本测试方法
# 以第一帧作为提示进行测试
# python tools/vos_inference_with_lora.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /opt/data/private/hyp/sam2/output_1224_sam2.1_hiera_l_hels_finetune+VOS+lora_with_new_loss/checkpoints/checkpoint_300.pt \
#     --base_video_dir /opt/data/private/lls/HLES-SAM/data/CVPR2027/video_test_1/JPEGImages \
#     --input_mask_dir /opt/data/private/lls/HLES-SAM/data/CVPR2027/video_test_1/Annotations \
#     --output_mask_dir /opt/data/private/lls/HLES-SAM/data/CVPR2027/video_test_1/Predictions \
#     --per_obj_png_file 


# 使用所有帧的输入进行测试
# python tools/vos_inference_with_lora.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /opt/data/private/hyp/sam2/output_1224_sam2.1_hiera_l_hels_finetune+VOS+lora_with_new_loss/checkpoints/checkpoint_300.pt \
#     --base_video_dir /opt/data/private/lls/HLES-SAM/data/CVPR2027/video_test_1/JPEGImages \
#     --input_mask_dir /opt/data/private/lls/HLES-SAM/data/CVPR2027/video_test_2/Annotations \
#     --output_mask_dir /opt/data/private/lls/HLES-SAM/data/CVPR2027/video_test_2/Predictions \
#     --per_obj_png_file  \
#     --use_all_masks


# 第一帧测试
# python tools/vos_inference_with_lora.py \
#     --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#     --sam2_checkpoint /opt/data/private/hyp/sam2/output_0105_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+small_area_penalty_numframes/checkpoints/checkpoint_400.pt \
#     --base_video_dir /opt/data/private/lls/HLES-SAM/data/CVPR2028/video_test_1/JPEGImages \
#     --input_mask_dir /opt/data/private/lls/HLES-SAM/data/CVPR2028/video_test_1/Annotations \
#     --output_mask_dir /opt/data/private/lls/HLES-SAM/data/CVPR2028/video_test_1/Predictions \
#     --per_obj_png_file 

# 所有帧测试
python tools/vos_inference_with_lora.py \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam2_checkpoint /opt/data/private/hyp/sam2/output_0105_sam2.1_hiera_l_hels_finetune+VOS+lora_+new_loss_+boundary_weight+small_area_penalty_numframes/checkpoints/checkpoint_390.pt \
    --base_video_dir /opt/data/private/lls/HLES-SAM/data/CVPR2028/video_test_1/JPEGImages \
    --input_mask_dir /opt/data/private/lls/HLES-SAM/data/CVPR2028/video_test_2/Annotations \
    --output_mask_dir /opt/data/private/lls/HLES-SAM/data/CVPR2028/video_test_2/Predictions \
    --per_obj_png_file  \
    --use_all_masks
