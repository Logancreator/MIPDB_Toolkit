#!/bin/bash
#SBATCH --job-name=PFLO_uav
#SBATCH --partition=gpu-v100s
#SBATCH --ntasks-per-node=4
#SBATCH --output=%j.log
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1

which python

CURDIR=`pwd`

python train_uav.py