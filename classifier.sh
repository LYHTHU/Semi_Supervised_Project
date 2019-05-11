#!/bin/bash
#SBATCH --job-name=WGCL-v1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --output=./WGCL-v1.out
#SBATCH --error=./WGCL-v1.err

python WGAN.py --learning_rate 3e-4 --dim_factor 128 --epochs 1



