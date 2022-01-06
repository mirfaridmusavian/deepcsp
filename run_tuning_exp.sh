#!/bin/sh
#SBATCH --job-name=2014004_3
#SBATCH --partition=htc
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40000M

module add python/anaconda3
module add gpu/cuda-11.3

conda init bash
source ~/.bashrc

conda activate bci

cd /gpfs/gpfs0/r.asiaban/projects/bci/deepcsp

python tuning.py
