# 1121 lora_tune
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_l_hels_finetune+lora_+new_loss_+dis5k_+boundary_weight.yaml \
    --use-cluster 0 \
    --num-gpus 2