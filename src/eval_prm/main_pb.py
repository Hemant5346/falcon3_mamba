from tqdm import tqdm
from datasets import load_dataset
from tabulate import tabulate
from copy import deepcopy
import os

import ray

from src.eval_utils.gen_utils import PRMPredictor


@ray.remote
class RemotePRMPredictor(PRMPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Step 0: Parameters

num_instances = 8
model_path = "/path/to/process_reward_model"
dataset_path = "Qwen/ProcessBench"
configs = ["gsm8k", "math", "olympiadbench", "omnimath"]


# Step 1: PRM Prediction & compute metrics
ray.init()
actors = [
    RemotePRMPredictor.options(num_gpus=1, num_cpus=1).remote(model_path=model_path)
    for _ in range(num_instances)
]


def compute_metrics(results, threshold):
    all_result = deepcopy(results)
    for result in all_result:
        result["pred_label"] = -1
        for idx, step_reward in enumerate(result["step_rewards"]):
            if step_reward < threshold:
                result["pred_label"] = idx
                break
        result["match"] = result["pred_label"] == result["label"]
    data1 = [e for e in all_result if e["label"] != -1]
    data2 = [e for e in all_result if e["label"] == -1]
    acc1 = sum([e["match"] for e in data1]) / len(data1)
    acc2 = sum([e["match"] for e in data2]) / len(data2)
    f1 = 2 * acc1 * acc2 / (acc1 + acc2)
    return {"err_acc": acc1, "cor_acc": acc2, "f1": f1}


eval_results = {}
for config in configs:
    eval_dataset = load_dataset(dataset_path, split=config)
    futures = ray.util.ActorPool(actors).map_unordered(
        lambda actor, sample: actor.score.remote(sample), eval_dataset
    )
    results = []
    for result in tqdm(futures):
        results.append(result)

    threshold = 0
    output_list = []
    while threshold < 1:
        metrics = compute_metrics(results, threshold)
        output_list.append(
            [threshold, metrics["err_acc"], metrics["cor_acc"], metrics["f1"]]
        )
        threshold += 0.05

    eval_results[config] = tabulate(
        output_list,
        headers=["threshold", "err_acc", "cor_acc", "f1"],
        tablefmt="grid",
    )

output_path = os.path.join(model_path, "eval_results")
os.makedirs(output_path, exist_ok=True)
with open(os.path.join(output_path, "process_bench.txt"), "w") as f:
    for dataset, result in eval_results.items():
        f.write(f"{dataset}:" + "\n\n")
        f.write(result + "\n\n\n")
