#!/bin/bash
#SBATCH --job-name=wgan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=./wgan.out
#SBATCH --error=./wgan.err

python WGAN.py --learning_rate 1e-4 --lmbda 10 --K 3 --noise_dim 128 --dim_factor 64 --epochs 2

