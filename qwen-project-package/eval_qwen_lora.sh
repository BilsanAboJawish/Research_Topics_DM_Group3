#!/bin/bash
#SBATCH --job-name=qwen-lora-eval
#SBATCH --account=gputue013
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch-shared/scur0448/qwen-research/logs/qwen-lora-eval-%j.out

set -e

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

source ~/venvs/qwen/bin/activate

export HF_HOME=/scratch-shared/$USER/qwen-research/hf_cache
export TOKENIZERS_PARALLELISM=false

cd ~/qwen-research/code

echo "Evaluating adapter: $1"

python -u scripts/evaluate_qwen_lora.py \
    --adapter "$1" \
    --output "$2"
