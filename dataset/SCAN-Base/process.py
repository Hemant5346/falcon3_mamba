import matplotlib.pyplot as plt
import orjson
import json
import os
from tqdm import tqdm
import numpy as np
import random


def load_jsonl(fpath: str, use_tqdm: bool = False) -> list:
    """Load JSONL file."""
    with open(fpath, "r") as f:
        lines: list[str] = f.readlines()
        return [
            orjson.loads(line)
            for line in (
                lines if not use_tqdm else tqdm(lines, desc=f"Loading {fpath}")
            )
        ]


def save_jsonl(data: list, fpath: str) -> None:
    """Save JSONL file."""
    with open(fpath, "w") as f:
        for line in data:
            f.write(orjson.dumps(line).decode() + "\n")


def get_data(data_path):
    datasets = load_jsonl(data_path)
    corr_list = []
    incorr_list = []

    for line in datasets:
        if line["generation_accuracy"] < 0.75:
            continue
        question = line["problem"]
        for generation in line["generation"]:
            steps = generation["steps"]
            if generation["correct"]:
                scores = [1.0] * len(steps)
                corr_list.append(
                    {
                        "question": question,
                        "steps": steps,
                        "scores": scores
                    }
                )
            else:

                if len(steps) == 1 and "steps_score" not in generation:
                    generation["steps_score"] = [0.0]

                temp_scores = generation["steps_score"]

                if 0.0 not in temp_scores:
                    continue

                first_err_loc = temp_scores.index(0.0)
                scores = [1] * first_err_loc + [0]

                if first_err_loc - 1 >= 0:
                    scores[first_err_loc - 1] = min(1, temp_scores[first_err_loc - 1] / line["generation_accuracy"])

                if first_err_loc - 2 >= 0:
                    scores[first_err_loc - 2] = min(1, temp_scores[first_err_loc - 2] / line["generation_accuracy"])

                incorr_list.append(
                    {
                        "question": question,
                        "steps": steps[: first_err_loc + 1],
                        "scores": scores[: first_err_loc + 1],
                    }
                )

    sample_num = min(len(corr_list), len(incorr_list))
    final_dataset = random.sample(corr_list, sample_num) + random.sample(incorr_list, sample_num)
    print(len(corr_list), len(incorr_list), len(final_dataset))
    return final_dataset


data = []
data += get_data("outputs/qwen25-15b-datasyn-math/samples_with_labels.jsonl")

output_path = "datasets/SCAN-Base"
print(len(data))
os.makedirs(output_path, exist_ok=True)
save_jsonl(data, os.path.join(output_path, "train.jsonl"))
