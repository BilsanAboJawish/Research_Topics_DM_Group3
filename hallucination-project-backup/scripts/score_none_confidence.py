import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TASKS = ["dilemma", "inter", "time"]


def args():
    p = argparse.ArgumentParser()

    p.add_argument("--name", required=True)
    p.add_argument("--result_root", required=True)
    p.add_argument("--marker", required=True)
    p.add_argument("--output", required=True)

    p.add_argument(
        "--adapter",
        default=None
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    return p.parse_args()


def find_file(root, marker, task, qtype):
    matches = [
        p for p in Path(root).rglob(
            f"samples_task1_{task}_en_{qtype}_*.jsonl"
        )
        if marker in str(p)
    ]

    if not matches:
        raise FileNotFoundError(
            f"{root}: {task} {qtype}"
        )

    matches.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return matches[0]


def load_samples(root, marker):
    rows = []

    for task in TASKS:
        for qtype in ["ab", "ua"]:

            path = find_file(
                root,
                marker,
                task,
                qtype
            )

            with open(path, encoding="utf-8") as f:
                for line in f:

                    if not line.strip():
                        continue

                    x = json.loads(line)

                    prompt = (
                        x["arguments"]
                        ["gen_args_0"]
                        ["arg_0"]
                    )

                    rows.append({
                        "task": task,
                        "qtype": qtype,
                        "doc_id": x["doc_id"],
                        "prompt": prompt,
                        "EM_hit":
                            float(x.get("EM_hit", 0)),
                        "NEC_refuse":
                            float(x.get("NEC_refuse", 0)),
                    })

    return rows


def score_batch(
    prompts,
    tokenizer,
    model,
    candidate_ids
):
    """
    Score log P(candidate | exact evaluation prompt).

    We use the exact stored UAQFact prompt and append
    tokenized 'None', without applying another chat template.
    """

    all_ids = []
    prompt_lengths = []

    for prompt in prompts:

        p_ids = tokenizer(
            prompt,
            add_special_tokens=False
        )["input_ids"]

        prompt_lengths.append(
            len(p_ids)
        )

        all_ids.append(
            p_ids + candidate_ids
        )

    max_len = max(
        len(x) for x in all_ids
    )

    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids = []
    attention_mask = []

    for ids in all_ids:

        padding = (
            [pad_id]
            * (max_len - len(ids))
        )

        input_ids.append(
            ids + padding
        )

        attention_mask.append(
            [1] * len(ids)
            + [0] * len(padding)
        )

    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=model.device
    )

    attention_mask = torch.tensor(
        attention_mask,
        dtype=torch.long,
        device=model.device
    )

    with torch.inference_mode():

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        log_probs = torch.log_softmax(
            output.logits.float(),
            dim=-1
        )

    scores = []

    for i, prompt_len in enumerate(
        prompt_lengths
    ):

        token_logps = []

        # Token k is predicted by logits at
        # position prompt_len + k - 1.
        for k, token_id in enumerate(
            candidate_ids
        ):

            position = (
                prompt_len + k - 1
            )

            token_logps.append(
                log_probs[
                    i,
                    position,
                    token_id
                ].item()
            )

        total = sum(token_logps)

        mean_lp = (
            total / len(token_logps)
        )

        scores.append(
            (total, mean_lp)
        )

    return scores


def main():

    a = args()

    samples = load_samples(
        a.result_root,
        a.marker
    )

    print(
        f"{a.name}: "
        f"loaded {len(samples)} examples"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    candidate_ids = tokenizer(
        "None",
        add_special_tokens=False
    )["input_ids"]

    print(
        "Refusal continuation: None"
    )

    print(
        "Token IDs:",
        candidate_ids
    )

    print(
        "Tokens:",
        tokenizer.convert_ids_to_tokens(
            candidate_ids
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    )

    if a.adapter:

        print(
            "Loading adapter:",
            a.adapter
        )

        model = PeftModel.from_pretrained(
            model,
            a.adapter
        )

    model.eval()

    results = []

    for start in range(
        0,
        len(samples),
        a.batch_size
    ):

        batch = samples[
            start:
            start + a.batch_size
        ]

        scores = score_batch(
            [x["prompt"] for x in batch],
            tokenizer,
            model,
            candidate_ids
        )

        for x, (
            total_lp,
            mean_lp
        ) in zip(
            batch,
            scores
        ):

            results.append({
                "model": a.name,
                "task": x["task"],
                "qtype": x["qtype"],
                "doc_id": x["doc_id"],

                "logp_none":
                    total_lp,

                "mean_logp_none":
                    mean_lp,

                "EM_hit":
                    x["EM_hit"],

                "NEC_refuse":
                    x["NEC_refuse"],
            })

        if start % 500 < a.batch_size:
            print(
                f"{min(start+a.batch_size,len(samples))}"
                f"/{len(samples)}"
            )

    output = Path(a.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("Saved:", output)


if __name__ == "__main__":
    main()
