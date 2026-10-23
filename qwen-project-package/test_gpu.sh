#!/bin/bash
#SBATCH --job-name=qwen-gpu-test
#SBATCH --account=gputue013
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/scratch-shared/scur0448/qwen-research/logs/gpu-test-%j.out

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

source ~/venvs/qwen/bin/activate

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
