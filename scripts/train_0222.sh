python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_l_hels_finetune+VOS+lora_+baseline_8_12.yaml \
    --use-cluster 0 \
    --num-gpus 8

python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_l_hels_finetune+VOS+lora_+baseline_12_8.yaml \
    --use-cluster 0 \
    --num-gpus 8

