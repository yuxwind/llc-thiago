#!/bin/sh

#SBATCH -o slurm-gpu-job.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH -c 4

sh scripts/train/CIFAR10-rgb/net5-run_1_1.sh > logs/CIFAR10-rgb/net3-run_1_1.log 2>&1
#sh scripts/train/CIFAR10-rgb/net5-run_2_2.sh > logs/CIFAR10-rgb/net3-run_2_2.log 2>&1
sh scripts/train/CIFAR10-rgb/net5-run_3_3.sh > logs/CIFAR10-rgb/net3-run_3_3.log 2>&1
sh scripts/train/CIFAR10-rgb/net5-run_4_4.sh > logs/CIFAR10-rgb/net3-run_4_4.log 2>&1
#sh scripts/train/CIFAR10-rgb/net5-run_5_5.sh > logs/CIFAR10-rgb/net3-run_5_5.log 2>&1
#sh train1_net3.sh > logs/train1_net3.log 2>&1
#sh train2_net3.sh > logs/train2_net3.log 2>&1
#sh train3_net3.sh > logs/train3_net3.log 2>&1
#sh train4_net3.sh > logs/train4_net3.log 2>&1
#sh train5_net3.sh > logs/train5_net3.log 2>&1
