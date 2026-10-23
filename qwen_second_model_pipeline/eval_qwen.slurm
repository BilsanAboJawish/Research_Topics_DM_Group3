#!/usr/bin/env bash
#SBATCH --job-name=qwen-eval
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
MODE="${MODE:?Set MODE=base|ifd|random}"
PROJECT="${PROJECT:-$HOME/hallucination-project}"
mkdir -p "$PROJECT/logs"

bash "$PROJECT/scripts/eval_qwen_uaqfact.sh" "$MODE"
