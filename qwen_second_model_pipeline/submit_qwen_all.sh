#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/hallucination-project}"
cd "$PROJECT"
mkdir -p logs results/qwen2p5_second_model models

# -------------------------------------------------------------------
# IMPORTANT: set these two paths to the EXACT SAME 15% subsets used
# in the Mistral experiment. Do NOT resample for Qwen.
# You may export them before running this file, or edit them here.
# -------------------------------------------------------------------
IFD_FILE="${IFD_FILE:-}"
RANDOM_FILE="${RANDOM_FILE:-}"

if [[ -z "$IFD_FILE" || -z "$RANDOM_FILE" ]]; then
  echo
  echo "Set IFD_FILE and RANDOM_FILE first."
  echo "Find likely files with:"
  echo "  find $PROJECT/data -type f | grep -Ei 'ifd.*15|15.*ifd|random.*15|15.*random'"
  echo
  echo "Then run, for example:"
  echo "  export IFD_FILE=/exact/path/to/ifd_15pct.json"
  echo "  export RANDOM_FILE=/exact/path/to/random_15pct.json"
  echo "  bash scripts/submit_qwen_all.sh"
  exit 2
fi

[[ -f "$IFD_FILE" ]] || { echo "Missing IFD_FILE: $IFD_FILE"; exit 3; }
[[ -f "$RANDOM_FILE" ]] || { echo "Missing RANDOM_FILE: $RANDOM_FILE"; exit 3; }

# Base evaluation can start immediately.
BASE_JOB=$(sbatch --parsable --export=ALL,MODE=base scripts/eval_qwen.slurm)
echo "Base eval job: $BASE_JOB"

# Train IFD and Random in parallel.
IFD_TRAIN=$(sbatch --parsable --export=ALL,COND=ifd,TRAIN_FILE="$IFD_FILE" scripts/train_qwen.slurm)
RND_TRAIN=$(sbatch --parsable --export=ALL,COND=random,TRAIN_FILE="$RANDOM_FILE" scripts/train_qwen.slurm)
echo "IFD train job: $IFD_TRAIN"
echo "Random train job: $RND_TRAIN"

# Each adapter evaluation starts automatically when its training succeeds.
IFD_EVAL=$(sbatch --parsable --dependency=afterok:$IFD_TRAIN --export=ALL,MODE=ifd scripts/eval_qwen.slurm)
RND_EVAL=$(sbatch --parsable --dependency=afterok:$RND_TRAIN --export=ALL,MODE=random scripts/eval_qwen.slurm)
echo "IFD eval job: $IFD_EVAL"
echo "Random eval job: $RND_EVAL"

# Final CPU summary waits for all three evaluations.
SUM_JOB=$(sbatch --parsable \
  --dependency=afterok:$BASE_JOB:$IFD_EVAL:$RND_EVAL \
  scripts/summarize_qwen.slurm)
echo "Summary job: $SUM_JOB"

echo
echo "All jobs submitted."
echo "Monitor with: squeue -u $USER"
echo "Results will be under: $PROJECT/results/qwen2p5_second_model/"
