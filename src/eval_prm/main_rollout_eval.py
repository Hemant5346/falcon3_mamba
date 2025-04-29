import os
import random
from tqdm import tqdm
from tabulate import tabulate

import torch
import ray

from src.eval_utils.data import load_datasets, load_jsonl, save_jsonl
from src.eval_utils.parser import parse_ground_truth, extract_and_strip
from src.eval_utils.grader import math_equal
from src.eval_utils.gen_utils import Generator


@ray.remote
class RemoteStepRolloutGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform_step_rollout(self, samples):
        prompts = []
        for sample in samples:
            question_prompt = self.prompt_func.make_full_prompt(sample["problem"])
            for step_idx in range(len(sample["steps"])):
                prompt = question_prompt + "\n\n".join(sample["steps"][:step_idx+1]) + "\n\n"
                prompts.extend([prompt] * self.n_sampling)
        outputs = self.llm.generate(prompts, self.sampling_params)
        idx = 0
        for sample in samples:
            step_rollouts = [[] for _ in range(len(sample["steps"]))]
            for step_idx in range(len(sample["steps"])):
                for _ in range(self.n_sampling):
                    step_rollouts[step_idx].append({"response": outputs[idx].outputs[0].text})
                    idx += 1
            sample["step_rollouts"] = step_rollouts
        return samples


@ray.remote
def remote_math_equal(pred, gt_ans):
    try:
        return math_equal(pred, gt_ans)
    except Exception:
        return False


@ray.remote
class RemoteRolloutEvaluator:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def evaluate_steps(self, sample):
        sample["gt_ans"] = parse_ground_truth(sample, sample["dataset"])[-1]
        step_rollout_scores = []
        for step_rollout in sample["step_rollouts"]:
            step_score = 0
            for rollout in step_rollout:
                rollout["pred"] = extract_and_strip(rollout["response"])
                try:
                    rollout["correct"] = ray.get(
                        remote_math_equal.remote(rollout["pred"], sample["gt_ans"]),
                        timeout=self.timeout
                    )
                except:
                    rollout["correct"] = False
                step_score += rollout["correct"]
            step_score /= len(step_rollout)
            step_rollout_scores.append(step_score)

        sample["step_rollout_scores"] = step_rollout_scores
        pred_label = -1
        for step_idx, step_score in enumerate(step_rollout_scores):
            if step_score == 0:
                pred_label = step_idx
                break

        sample["pred_label"] = pred_label
        return sample


def compute_noise_transition(samples):
    counts = {
        "pred = label = Correct": 0,
        "pred = label = Error": 0,
        "pred pos < label pos": 0,
        "pred pos > label pos": 0,
    }
    for sample in samples:
        pred, label = sample["pred_label"], sample["label"]
        if pred == -1:
            pred = len(sample["steps"])
        if label == -1:
            label = len(sample["steps"])
        if pred == len(sample["steps"]) and label == len(sample["steps"]):
            counts["pred = label = Correct"] += 1
        elif pred == label:
            counts["pred = label = Error"] += 1
        elif pred < label:
            counts["pred pos < label pos"] += 1
        else:
            counts["pred pos > label pos"] += 1
    return counts


# Step 0: Parameters
ray.init()

num_instances = torch.cuda.device_count()
lm_args = {
    "model_path": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "prompt_template": "qwen25-math-cot",
    "num_sequence": 8,
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
    "tensor_parallel_size": 1,
}
datasets = [
    "process_bench/gsm8k",
    "process_bench/math",
    "process_bench/olympiadbench",
    "process_bench/omnimath",
]
generate = True
evaluate = False
analyze = True
output_dir = "outputs/qwen25-15b-processbench-rollouts"
os.makedirs(output_dir, exist_ok=True)

# Step 1: load datasets
datasets = load_datasets(datasets)
random.shuffle(datasets)

# Step 2: LM Generation
lm_output_path = os.path.join(output_dir, "generations.jsonl")
if generate:
    batch_size = min(10000, len(datasets) // num_instances + 1)
    llm_actors = [
        RemoteStepRolloutGenerator.options(num_gpus=1, num_cpus=4).remote(**lm_args)
        for _ in range(num_instances)
    ]
    dataset_batches = [
        datasets[i : i + batch_size] for i in range(0, len(datasets), batch_size)
    ]
    futures = ray.util.ActorPool(llm_actors).map_unordered(
        lambda actor, batch_data: actor.perform_step_rollout.remote(batch_data),
        dataset_batches,
    )
    completed_futures = []
    for batch_idx, batch_results in enumerate(futures):
        completed_futures.extend(batch_results)
    save_jsonl(completed_futures, lm_output_path)


# Step 3: Evaluating
eval_output_path = os.path.join(output_dir, "eval_results.jsonl")
if evaluate:
    datasets = load_jsonl(lm_output_path)
    num_cpus = 16
    evaluators = [
        RemoteRolloutEvaluator.options(num_cpus=1).remote() for _ in range(num_cpus)
    ]
    futures = ray.util.ActorPool(evaluators).map_unordered(
        lambda actor, sample: actor.evaluate_steps.remote(sample), datasets
    )
    results = []
    for result in tqdm(futures, total=len(datasets), desc="Evaluating samples"):
        results.append(result)
    save_jsonl(results, eval_output_path)

# Step 4: Analyze
metric_output_path = os.path.join(output_dir, "metrics.txt")
if analyze:
    datasets = load_jsonl(eval_output_path)
    eval_results = {}
    for config in ["gsm8k", "math", "olympiadbench", "omnimath"]:
        subset = [data_point for data_point in datasets if data_point["dataset"] == config]
        counts = compute_noise_transition(samples=subset)
        total = sum(counts.values())
        data = [
            [category, count, f"{(count / total) * 100:.2f}%"]
            for category, count in counts.items()
        ]
        headers = ["Category", "Count", "Percentage"]
        eval_results[config] = tabulate(data, headers, tablefmt="grid")

    with open(metric_output_path, "w") as f:
        for dataset, result in eval_results.items():
            f.write(f"{dataset}:" + "\n\n")
            f.write(result + "\n\n\n")