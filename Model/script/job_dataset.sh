#!/bin/bash
#SBATCH -J dataset
#SBATCH -p AMD_9A14,AMD_9654
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH -o dataset_%A_%a.out

source /public/home/changjianye/miniconda3/etc/profile.d/conda.sh
conda activate phenomics

cd /public/share/ruanjue/wangmengtao/Project/keypoint/Data

# dslr
python 02.labelme2yolo_resize.py --json_dir /public/share/ruanjue/wangmengtao/Project/keypoint/Data/DSLR/label_17 \
--output_dir /public/share/ruanjue/wangmengtao/Project/keypoint/Data/resize_dslr \
--image_dir /public/share/ruanjue/wangmengtao/Project/keypoint/Data/DSLR/image \
--resize 4