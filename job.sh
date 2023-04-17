#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --gres=gpu:t4:2
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --account=def-dzhaomac
#SBATCH --output=slurm.single.%x.%j.out

python Image_Captioning.py
