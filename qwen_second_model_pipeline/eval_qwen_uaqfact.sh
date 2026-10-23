#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: eval_qwen_uaqfact.sh base|ifd|random}"
PROJECT="${PROJECT:-$HOME/hallucination-project}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
UAQ_ROOT="${UAQ_ROOT:-$PROJECT/UAQ_Fact}"
RESULT_ROOT="$PROJECT/results/qwen2p5_second_model"

case "$MODE" in
  base)
    MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16"
    OUT="$RESULT_ROOT/base"
    ;;
  ifd)
    ADAPTER="$PROJECT/models/qwen2p5_lora_ifd_15pct_seed42"
    MODEL_ARGS="pretrained=$MODEL,peft=$ADAPTER,dtype=bfloat16"
    OUT="$RESULT_ROOT/ifd15"
    ;;
  random)
    ADAPTER="$PROJECT/models/qwen2p5_lora_random_15pct_seed42"
    MODEL_ARGS="pretrained=$MODEL,peft=$ADAPTER,dtype=bfloat16"
    OUT="$RESULT_ROOT/random15"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ ! -d "$UAQ_ROOT" ]]; then
  echo "UAQFact repo not found at $UAQ_ROOT" >&2
  echo "Clone it first: cd $PROJECT && git clone https://github.com/cytan17726/UAQ_Fact.git" >&2
  exit 3
fi

source "$HOME/venvs/uaqfact_eval/bin/activate"

# Discover the exact English Task-1 task names from the official YAML files.
TASKS=$(
  grep -RhsE '^[[:space:]]*task:[[:space:]]*' "$UAQ_ROOT/task/uaq_fact" \
    | sed -E 's/^[[:space:]]*task:[[:space:]]*//; s/["'\'']//g' \
    | grep -E 'task1_.*_en_(ab|ua)$' \
    | sort -u \
    | paste -sd, -
)

if [[ -z "${TASKS}" ]]; then
  echo "Could not auto-discover English Task-1 UAQFact tasks." >&2
  echo "Inspect with: grep -R '^task:' $UAQ_ROOT/task/uaq_fact" >&2
  exit 4
fi

mkdir -p "$OUT"
echo "Mode: $MODE"
echo "Tasks: $TASKS"
echo "Output: $OUT"

cd "$UAQ_ROOT/lm-evaluation-harness"

python -m lm_eval \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --include_path "$UAQ_ROOT/task" \
  --device cuda:0 \
  --batch_size auto \
  --apply_chat_template \
  --log_samples \
  --output_path "$OUT"

echo "EVALUATION DONE: $OUT"
