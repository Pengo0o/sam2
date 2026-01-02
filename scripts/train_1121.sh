# 1121 lora_tune
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_l_hels_finetune+lora_with_new_loss_with_new_data.yaml \
    --use-cluster 0 \
    --num-gpus 2