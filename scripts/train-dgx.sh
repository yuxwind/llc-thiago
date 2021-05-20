#!/bin/sh

#SBATCH -o slurm-gpu-job.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -c 8

sh train1_net6.sh > logs/train1_net6.log 2>&1
#sh train2_net6.sh > logs/train2_net6.log 2>&1
#sh train3_net6.sh > logs/train3_net6.log 2>&1
#sh train4_net6.sh > logs/train4_net6.log 2>&1
#sh train5_net6.sh > logs/train5_net6.log 2>&1
