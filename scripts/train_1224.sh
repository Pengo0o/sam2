# 1205 lora_tune
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_l_hels_finetune+VOS+lora_with_new_loss.yaml \
    --use-cluster 0 \
    --num-gpus 2