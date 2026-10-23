#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-$HOME/hallucination-project}"

mkdir -p "$PROJECT/scripts" "$PROJECT/logs" "$PROJECT/models" "$PROJECT/results/qwen2p5_second_model"

echo "1) Qwen compatibility check"
source "$HOME/venvs/hallucination1/bin/activate"
python - <<'PY'
import transformers, torch, peft, datasets
print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("peft:", peft.__version__)
print("datasets:", datasets.__version__)
from packaging.version import Version
assert Version(transformers.__version__) >= Version("4.37.0")
PY
deactivate || true

echo
echo "2) UAQFact official repository"
if [[ ! -d "$PROJECT/UAQ_Fact" ]]; then
  cd "$PROJECT"
  git clone https://github.com/cytan17726/UAQ_Fact.git
fi

echo
echo "3) Install the official UAQFact lm-eval fork into the existing eval venv"
source "$HOME/venvs/uaqfact_eval/bin/activate"
cd "$PROJECT/UAQ_Fact/lm-evaluation-harness"
pip install -e .
deactivate || true

echo
echo "SETUP COMPLETE"
