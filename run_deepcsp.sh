#!/bin/sh
#SBATCH --job-name=eegent
#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40000M


module add python/anaconda3
module add gpu/cuda-11.3
module add intel/mkl/2020
module add lib/magma-2.5.3
module add compilers/gcc-8.3.0 
module add compilers/glibc-2.18

module unload gpu/cuda-10.2
module unload intel/mkl/2019

source activate /gpfs/gpfs0/m.musavian/conda_envs/bci

cd /gpfs/gpfs0/m.musavian/bci/deepcsp

python deepcsp.py
