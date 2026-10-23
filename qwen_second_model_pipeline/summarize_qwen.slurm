#!/usr/bin/env bash
#SBATCH --job-name=qwen-summary
#SBATCH --partition=thin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT="${PROJECT:-$HOME/hallucination-project}"
OUT="$PROJECT/results/qwen2p5_second_model"
mkdir -p "$OUT"

for mode in base ifd15 random15; do
  python "$PROJECT/scripts/summarize_qwen_uaqfact.py" \
    --root "$OUT/$mode" \
    --name "$mode" \
    > "$OUT/${mode}_summary.txt"
done

cat "$OUT/base_summary.txt"
cat "$OUT/ifd15_summary.txt"
cat "$OUT/random15_summary.txt"
