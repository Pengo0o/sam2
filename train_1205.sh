# 1205 lora_tune
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_l_hels_finetune+lora_+new_loss_+dis5k_+boundary_weight+optimal_mask.yaml \
    --use-cluster 0 \
    --num-gpus 2