#!/bin/bash
#SBATCH --job-name=qwen-ifd05
#SBATCH --account=gputue013
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/scur0448/qwen-research/logs/qwen-ifd05-%j.out

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

source ~/venvs/qwen/bin/activate

export HF_HOME=/scratch-shared/$USER/qwen-research/hf_cache
export TOKENIZERS_PARALLELISM=false

cd ~/qwen-research/code || exit 1

echo "Node: $(hostname)"
echo "Starting Qwen 5% IFD training"
nvidia-smi

python scripts/train_qwen_05pct_seed2026.py
