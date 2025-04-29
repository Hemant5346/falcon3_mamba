import os
import json
from typing import Any
from datasets import load_dataset
import orjson
from tqdm import tqdm


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


def load_json(fpath: str) -> dict:
    """Load JSON file."""
    with open(fpath, "r") as f:
        return orjson.loads(f.read())


def save_json(data: dict, fpath: str, indent: int = 4) -> None:
    """Save JSON file."""
    with open(fpath, "w") as f:
        json.dump(data, f, indent=indent)


def read_jsonl_dir(directory_path):
    file_contents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json") or filename.endswith(".jsonl"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = [json.loads(line) for line in f]
                file_contents.extend(content)
    return file_contents


def load_datasets(datasets: list[str] = ["gsm8k/test"]) -> list[dict]:
    data_pool = []
    for dataset in datasets:
        if os.path.exists(dataset):
            data_points = load_dataset(dataset, split="train")
            for data_pool in data_points:
                pass
            data_pool.extend(data_points)
        elif dataset.startswith("gsm8k/"):
            split = dataset.split("/")[-1]
            data_points = load_dataset("eval_data/gsm8k", split=split)
            data_points = data_points.map(
                lambda x: {**x, "dataset": "gsm8k"}
            ).rename_column("question", "problem")
            data_pool.extend(data_points)
        elif dataset.startswith("math/"):
            split = dataset.split("/")[-1]
            data_points = load_dataset("eval_data/math", split=split)
            data_points = data_points.map(lambda x: {**x, "dataset": "math"})
            data_pool.extend(data_points)
        elif dataset.startswith("math500/"):
            split = dataset.split("/")[-1]
            data_points = load_dataset("eval_data/math500", split=split)
            data_points = data_points.map(lambda x: {**x, "dataset": "math500"})
            data_pool.extend(data_points)
        elif dataset.startswith("college_math/"):
            split = dataset.split("/")[-1]
            data_points = load_dataset("eval_data/college_math", split=split)
            data_points = data_points.map(
                lambda x: {**x, "dataset": "college_math"}
            ).rename_column("question", "problem")
            data_pool.extend(data_points)
        elif dataset.startswith("minerva_math/"):
            split = dataset.split("/")[-1]
            data_points = load_dataset("eval_data/minerva_math", split=split)
            data_points = data_points.map(lambda x: {**x, "dataset": "minerva_math"})
            data_pool.extend(data_points)
        elif dataset.startswith("olympiadbench/"):
            split = dataset.split("/")[-1]
            data_points = load_dataset("eval_data/olympiadbench", split=split)
            data_points = data_points.map(
                lambda x: {**x, "dataset": "olympiadbench"}
            ).rename_column("question", "problem")
            data_pool.extend(data_points)
        elif dataset.startswith("numina_math/"):
            subet_name = dataset.split("/")[-1]
            dataset_name = f"numina_math_{subet_name}"
            data_points = load_dataset(f"datasets/NuminaMath-CoT", split="train")
            data_points = data_points.filter(
                lambda x: x["source"] == subet_name, num_proc=8
            )
            data_points = data_points.select_columns(["problem", "solution"])
            data_points = data_points.map(
                lambda x: {**x, "dataset": dataset_name}, num_proc=8
            )
            data_pool.extend(data_points)
        elif dataset.startswith("aime24/"):
            split = dataset.split("/")[-1]
            data_points = load_dataset("eval_data/aime24", split=split)
            data_points = data_points.map(lambda x: {**x, "dataset": "aime24"})
            data_pool.extend(data_points)
        elif dataset.startswith("amc23/"):
            split = dataset.split("/")[-1]
            data_points = load_dataset("eval_data/amc23", split=split)
            data_points = data_points.map(lambda x: {**x, "dataset": "amc23"})
            data_pool.extend(data_points)
        elif dataset.startswith("process_bench/"):
            split = dataset.split("/")[-1]
            data_points = load_dataset("eval_data/ProcessBench", split=split)
            if split == "gsm8k":
                gsm8k_dataset = load_dataset("eval_data/gsm8k", split="test")
                question2solution = {
                    dp["question"]: dp["answer"] for dp in gsm8k_dataset
                }
                data_points = data_points.map(
                    lambda x: {
                        **x,
                        "answer": question2solution[x["problem"]],
                        "dataset": "gsm8k",
                    }
                )
                data_pool.extend(data_points)
            elif split == "math":
                math_dataset = load_dataset("eval_data/math", split="test")
                question2solution = {
                    dp["problem"]: dp["solution"] for dp in math_dataset
                }
                data_points = data_points.map(
                    lambda x: {
                        **x,
                        "solution": question2solution[x["problem"]],
                        "dataset": "math",
                    }
                )
                data_pool.extend(data_points)
            elif split == "olympiadbench":
                olympiad_dataset = load_dataset("eval_data/olympiadbench", split="test")
                question2answer = {
                    dp["question"].replace("$\\quad$ ", ""): dp["final_answer"] for dp in olympiad_dataset
                }
                data_points = data_points.map(
                    lambda x: {
                        **x,
                        "final_answer": question2answer[x["problem"]],
                        "dataset": "olympiadbench",
                    }
                )
                data_pool.extend(data_points)
            elif split == "omnimath":
                omnimath_dataset = load_dataset("eval_data/omnimath", split="test")
                question2solution = {
                    dp["problem"]: dp["solution"] for dp in omnimath_dataset
                }
                data_points = data_points.map(
                    lambda x: {
                        **x,
                        "solution": question2solution[x["problem"]],
                        "dataset": "omnimath",
                    }
                )
                data_pool.extend(data_points)
            else:
                raise NotImplementedError(f"Split {split} not implemented for {dataset}")
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")

    return data_pool
