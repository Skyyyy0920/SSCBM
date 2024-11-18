#!/bin/bash
#SBATCH --job-name=glem-arxiv          # 作业名称
#SBATCH --ntasks=1                  # 任务数
#SBATCH --cpus-per-task=10          # 每个任务的 CPU 核心数
#SBATCH --mem=64G                    # 内存大小
#SBATCH --gres=gpu:v100:1           # 请求 1 个 V100 GPU
#SBATCH --time=72:00:00              # 运行时间限制 (hh:mm:ss)
#SBATCH --output=my_job_%j.log      # 标准输出和错误日志文件


python main.py --dataset CUB-200-2011