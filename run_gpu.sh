#!/bin/bash
#SBATCH --job-name=wgan-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=./wgan-4.out
#SBATCH --error=./wgan-4.err

python WGAN.py --learning_rate 1e-4 --lmbda 10 --K 4 --noise_dim 128 --dim_factor 64 --epochs 7

