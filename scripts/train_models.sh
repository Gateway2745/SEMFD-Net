#!/bin/bash

declare -a models=("vgg16" "densenet121" "resnet50" "resnet101" "resnest50" "resnest101")

if [[ "$1" == "uncropped" ]]
then 
    for i in "${models[@]}"
    do
        echo "using model $i"
        python train.py --train_path ./PlantDoc-Dataset/train --val_path ./PlantDoc-Dataset/val --test_path ./PlantDoc-Dataset/test --batch_size 16 --model "$i" --gpus 3 --snapshot_path models_b_16_"$i"_uncropped --tensorboard_dir tb_b_16_"$i"_uncropped --show_every 10 --epochs 350  --lr 0.000005 
    done
elif  [[ "$1" == "cropped" ]]
then 
    for i in "${models[@]}"
    do
        echo "using model $i"
        python train.py --train_path ./PD-C/train --val_path ./PD-C/val --test_path ./PD-C/test --batch_size 16 --model "$i" --gpus 3 --snapshot_path models_b_16_"$i"_cropped --tensorboard_dir tb_b_16_"$i"_cropped --show_every 10 --epochs 350  --lr 0.000005 
    done
else
    echo "invalid command"
fi