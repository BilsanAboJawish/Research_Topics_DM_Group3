#!/usr/bin/env bash
#SBATCH --job-name=qwen-train
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
COND="${COND:?Set COND=ifd or random}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to the exact 15% subset file}"

PROJECT="${PROJECT:-$HOME/hallucination-project}"
mkdir -p "$PROJECT/logs" "$PROJECT/models"

source "$HOME/venvs/hallucination1/bin/activate"

python - <<'PY'
import transformers, torch
print("transformers", transformers.__version__)
print("torch", torch.__version__)
from packaging.version import Version
assert Version(transformers.__version__) >= Version("4.37.0"), "Qwen2 requires transformers >=4.37"
PY

if [[ "$COND" == "ifd" ]]; then
  OUT="$PROJECT/models/qwen2p5_lora_ifd_15pct_seed42"
elif [[ "$COND" == "random" ]]; then
  OUT="$PROJECT/models/qwen2p5_lora_random_15pct_seed42"
else
  echo "Bad COND=$COND" >&2
  exit 2
fi

python "$PROJECT/scripts/train_qwen_lora.py" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUT" \
  --seed 42
