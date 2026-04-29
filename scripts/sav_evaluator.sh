# CVPR2027
# python sav_dataset/sav_evaluator.py \
#     --gt_root /opt/data/private/lls/HLES-SAM/data/CVPR2027/video_test_1/Annotations \
#     --pred_root /opt/data/private/lls/HLES-SAM/data/CVPR2027/video_test_2/Predictions \
#     --strict


# CVPR2028
# python sav_dataset/sav_evaluator.py \
#     --gt_root /opt/data/private/lls/HLES-SAM/data/CVPR2028/video_test_1/Annotations \
#     --pred_root /opt/data/private/lls/HLES-SAM/data/CVPR2028/video_test_1/Predictions \
#     --strict

python sav_dataset/sav_evaluator_with_boundary_iou.py \
    --gt_root /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Annotations \
    --pred_root /root/workspace/d663ovsp420c73cg8l00/data/CVPR2029/test/video/Predictions_0218_fusion \
    --strict \
    --do_not_skip_first_and_last_frame






    
