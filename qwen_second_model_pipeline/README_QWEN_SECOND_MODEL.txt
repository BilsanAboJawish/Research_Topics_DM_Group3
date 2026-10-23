QWEN SECOND-MODEL REPLICATION — FAST PATH
=============================================

MODEL
-----
Qwen/Qwen2.5-7B-Instruct

Why this model:
- ~7.6B parameters: comparable scale to Mistral-7B.
- Different model family from Mistral.
- Han & Huang's assigned paper reports Llama3.1-8B and Mistral-7B-v0.3,
  not Qwen2.5, so this is a useful cross-family replication.
- Qwen2.5 is supported by Transformers >=4.37 and has a native chat template.
- UAQFact itself evaluates a Qwen2.5 model, so the benchmark/model combination
  is already known to be meaningful.

SCOPE
-----
Do NOT repeat the whole Mistral matrix now.
Run only:
  1. Qwen base
  2. Qwen + IFD 15% (same exact 7,800-example subset used for Mistral)
  3. Qwen + Random 15% (same exact 7,800-example subset used for Mistral)

This directly tests whether the central IFD-vs-Random refusal pattern survives
outside Mistral.

IMPORTANT FAIRNESS RULE
-----------------------
Reuse the exact same IFD-15% and Random-15% DATA EXAMPLES.
Do not resample the Random subset.
Use Qwen's native chat template for training/evaluation, because it is an
instruction-tuned Qwen checkpoint.

FILES TO COPY INTO ~/hallucination-project/scripts/
----------------------------------------------------
train_qwen_lora.py
eval_qwen_uaqfact.sh
summarize_qwen_uaqfact.py
train_qwen.slurm
eval_qwen.slurm
summarize_qwen.slurm
submit_qwen_all.sh
setup_qwen_second_model.sh

FIRST: FIND YOUR EXISTING 15% SUBSETS
-------------------------------------
On Snellius:
  cd ~/hallucination-project
  find data -type f | grep -Ei 'ifd.*15|15.*ifd|random.*15|15.*random'

Identify the exact files used by your current Mistral 15% scripts.
If uncertain, inspect those scripts:
  grep -RniE '15pct|7800|train.*file|subset' scripts/train_lora_15pct.py scripts/train_lora_random_15pct.py

SECOND: SETUP ONCE
------------------
  cd ~/hallucination-project
  chmod +x scripts/*.sh scripts/*.slurm
  bash scripts/setup_qwen_second_model.sh

THIRD: SUBMIT EVERYTHING TOGETHER
---------------------------------
  export IFD_FILE=/exact/path/to/the/current/IFD_15pct_subset.json
  export RANDOM_FILE=/exact/path/to/the/current/random_15pct_subset.json
  bash scripts/submit_qwen_all.sh

This submits:
- Base UAQFact evaluation immediately.
- IFD training and Random training in parallel.
- IFD/Random evaluations automatically after their corresponding training.
- A summary job after all evaluations finish.

Monitor:
  squeue -u $USER

Check logs:
  tail -f logs/qwen-train-<JOBID>.out
  tail -f logs/qwen-eval-<JOBID>.out

Final results:
  ~/hallucination-project/results/qwen2p5_second_model/

SANITY CHECK
------------
The official UAQFact paper reports a Qwen2.5 baseline, so your base behavior
should be in a broadly plausible range. Do not expect exact equality because
checkpoint/version, chat formatting, and environment can differ.

IF SOMETHING FAILS
------------------
1. Training KeyError qwen2:
   transformers is too old. Qwen2.5 requires >=4.37.
2. UAQ tasks not found:
   run:
     grep -R '^task:' ~/hallucination-project/UAQ_Fact/task/uaq_fact
   and send the output.
3. OOM:
   reduce evaluation batch from --batch_size auto to --batch_size 1.
   Training is designed for one A100 and batch=1 with gradient accumulation.
4. Wrong data fields:
   train_qwen_lora.py prints dataset columns and will fail clearly.
   Send the columns/output and adjust the small infer_fields function.
5. Do not silently switch datasets or regenerate Random data.
