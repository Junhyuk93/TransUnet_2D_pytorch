#!/bin/bash

# Specify the folder containing the images
image_folder="/mnt/dataset/pmct/jh_dataset/test/image/2022-M-848"

# Specify the model path
model_path="/mnt/home/jhpark1/node3.gpu/transunet_pytorch/test_pmct.pth"

# Specify the output folder
output_folder="./results/transunet_result"

# Create the output folder if it doesn't exist
mkdir -p $output_folder

# Iterate over all images in the folder
for image_path in "$image_folder"/*; do
    # Run the inference command for each image
    CUDA_VISIBLE_DEVICES=3 python main.py --mode inference --model_path "$model_path" --image_path "$image_path"
done

