import os
import sys
import argparse
from tqdm import tqdm
from collections import defaultdict
from tabulate import tabulate
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Add root to sys.path (for src imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.eval_utils.data import load_datasets, save_jsonl, load_jsonl
from src.eval_utils.grader import math_equal
from src.eval_utils.parser import parse_ground_truth, extract_and_strip
from src.eval_utils.vote import AGG_FN_MAP


def compute_metrics_fn(eval_results, k, agg_method):
    final_results = []
    for sample in eval_results:
        eval_samples = [
            {
                "dataset": sample["dataset"],
                "ans": generation["pred"],
                "scores": generation.get("step_rewards", None),
                "correct": generation["correct"],
            }
            for generation in sample["generation"][:k]
        ]
        final_result = AGG_FN_MAP[agg_method](eval_samples)
        final_results.append(final_result)

    dataset_counts = defaultdict(int)
    dataset_correct = defaultdict(int)
    for result in final_results:
        dataset = result["dataset"]
        dataset_counts[dataset] += 1
        if result["correct"]:
            dataset_correct[dataset] += 1

    metrics = []
    for dataset, count in dataset_counts.items():
        correct = dataset_correct[dataset]
        metrics.append({
            "dataset": dataset,
            "total": count,
            "correct": correct,
            "accuracy": correct / count
        })

    average_accuracy = np.mean([float(metric["accuracy"]) for metric in metrics]) if metrics else 0.0
    metrics.append({"Average": average_accuracy})
    return metrics


# === CLI Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default="tiiuae/falcon-rw-1b-mamba")
parser.add_argument('--prompt_type', type=str, default="falcon-mamba-math-cot")
parser.add_argument('--data_name', type=str, default="college_math")
parser.add_argument('--split', type=str, default="test")
parser.add_argument('--output_dir', type=str, default="./outputs/falcon_3_mamba")
parser.add_argument('--num_test_sample', type=int, default=0)
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.8)
parser.add_argument('--save_outputs', action="store_true")
args = parser.parse_args()

output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)
print(f"[DEBUG] Using output directory: {output_dir}")

# === Load model & tokenizer ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# === Load datasets ===
datasets = load_datasets([f"{args.data_name}/{args.split}"])
if args.num_test_sample:
    datasets = datasets[:args.num_test_sample]

print(f"[DEBUG] First sample:\n{datasets[0]}")
print(f"[DEBUG] Available keys: {list(datasets[0].keys())}")

# === Generate answers ===
generations = []
for sample in tqdm(datasets, desc="Generating responses"):
    prompt = next((sample.get(k) for k in ["question", "prompt", "input", "problem"] if sample.get(k)), None)
    if not prompt:
        print(f"[WARNING] Skipping malformed sample with keys: {list(sample.keys())}")
        continue

    try:
        output = qa_pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=False,
            temperature=args.temperature,
            top_p=args.top_p
        )[0]["generated_text"]
    except Exception as e:
        print(f"[ERROR] Pipeline generation failed for prompt: {prompt} — {e}")
        continue

    generations.append({
        "dataset": args.data_name,
        "question": prompt,
        "response": output,
        "pred": extract_and_strip(output, data_name=args.data_name),
        "gt_ans": sample.get("answer") or parse_ground_truth(sample, args.data_name)[-1]
    })

if args.save_outputs:
    save_jsonl(generations, os.path.join(output_dir, "generations.jsonl"))

# === Evaluate ===
for gen in generations:
    try:
        gen["correct"] = math_equal(gen["pred"], gen["gt_ans"])
    except Exception as e:
        print(f"[ERROR] Failed to evaluate: {gen['pred']} vs {gen['gt_ans']} – {e}")
        gen["correct"] = False

save_jsonl(generations, os.path.join(output_dir, "eval_results.jsonl"))

# === Compute metrics ===
metrics_output_path = os.path.join(output_dir, "metrics.txt")
eval_results = []
for g in generations:
    eval_results.append({
        "dataset": g["dataset"],
        "generation": [{"pred": g["pred"], "correct": g["correct"]}]
    })

agg_fn_list = ["pass", "majority_vote"]
all_results = {}
for agg_method in agg_fn_list:
    metrics = compute_metrics_fn(eval_results, k=1, agg_method=agg_method)
    result = {"k": 1}
    result.update({metric["dataset"]: metric["accuracy"] for metric in metrics[:-1]})
    result.update(metrics[-1])
    all_results[agg_method] = [result]

with open(metrics_output_path, "w") as f:
    for agg_method, result in all_results.items():
        f.write(f"{agg_method}:\n\n")
        f.write(tabulate(result, headers="keys", tablefmt="grid", floatfmt=".4f") + "\n\n\n")
