#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --partition=gpu-3090
#SBATCH --ntasks-per-node=8
#SBATCH --output=%j.log
#SBATCH --nodelist=gpu5
#SBATCH --gres=gpu:1

which python

CURDIR=`pwd`


python predict.py


# python viz_and_metrics.py