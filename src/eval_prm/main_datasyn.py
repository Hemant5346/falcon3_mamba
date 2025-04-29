import os
from tqdm import tqdm, trange
import random
import time
import concurrent.futures

import torch
import ray

from src.eval_utils.data import load_datasets, load_jsonl, save_jsonl, save_json
from src.eval_utils.grader import math_equal
from src.eval_utils.parser import parse_ground_truth, extract_and_strip
from src.eval_utils.gen_utils import Generator, split_steps


@ray.remote
class RemoteGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@ray.remote
def remote_math_equal(pred, gt_ans):
    return math_equal(pred, gt_ans)


def init_request_list(samples):
    request_list = []
    for sample in samples:
        for generation in sample["generation"]:
            generation["steps"] = split_steps(generation["response"])
            generation["search_process"] = []
            if generation["correct"]:
                generation["pred_label"] = -1
            elif len(generation["steps"]) == 1:
                generation["pred_label"] = 0
            else:
                generation["steps_score"] = [-1] * len(generation["steps"])
                request_item = {
                    "problem": sample["problem"],
                    "gt_ans": sample["gt_ans"],
                    "generation": generation,
                    "step_idx": 0,
                }
                request_list.append(request_item)
    return request_list

@ray.remote(num_gpus=1)
class RemoteStepSearchRequestGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_rollout(self, request_list):
        num_rollout = self.n_sampling
        prompts = []
        gt_ans_list = []
        
        for request in request_list:
            question_prompt = self.prompt_func.make_full_prompt(request["problem"])
            step_idx = request["step_idx"]
            pre_steps = request["generation"]["steps"][: step_idx + 1]
            prefix_solution = "\n\n".join(pre_steps) + "\n\n"
            prompt = [question_prompt + prefix_solution for _ in range(num_rollout)]
            prompts.extend(prompt)
            gt_ans_list.extend([request["gt_ans"]] * len(prompt))

        outputs = self.generate(prompts)
        outputs_str = [output.outputs[0].text for output in outputs]
        # outputs_str = [request["generation"]["response"] for request in request_list for _ in range(num_rollout)]
        preds_list = [extract_and_strip(rollout) for rollout in outputs_str]
        return outputs_str, preds_list, gt_ans_list


@ray.remote
class RemoteStepSearchRequestEvaluator:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def evaluate(self, pred, gt_ans):
        eval_result = _evaluate_math_equal(pred, gt_ans)
        return eval_result


def update_request_list(num_rollout, request_list, rollout_list):
    new_request_list = []

    for idx, request in enumerate(request_list):
        sample_rollouts = rollout_list[idx * num_rollout : (idx + 1) * num_rollout]
        rollout_score = sum([rollout["correct"] for rollout in sample_rollouts]) / num_rollout
        generation = request["generation"]
        step_idx = request["step_idx"]
        generation["steps_score"][step_idx] = rollout_score
        generation["search_process"].append(
            {"step_idx": step_idx, "rollout_score": rollout_score, "rollout_list": sample_rollouts}
        )
        if rollout_score == 0 or step_idx == len(generation["steps"]) - 1:  # last step
            generation["pred_label"] = step_idx
        else:
            request["step_idx"] += 1
            new_request_list.append(request)

    return new_request_list


# Step 0: Parameters
lm_args = {
    "model_path": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "prompt_template": "qwen25-math-cot",
    "num_sequence": 128,
    "max_tokens": 2048,
    "temperature": 1.0,
    "top_p": 1.0,
    "tensor_parallel_size": 1,
}
num_rollouts = 8
output_dir = "outputs/scan-qwen-15b"
datasets = [
    # "gsm8k/train",
    "math/train",
    # "numina_math/aops_forum",
    # "numina_math/amc_aime",
    # "numina_math/cn_k12",
    # "numina_math/gsm8k",
    # "numina_math/math",
    # "numina_math/olympiads",
    # "numina_math/orca_math",
    # "numina_math/synthetic_amc",
    # "numina_math/synthetic_math",
]

os.makedirs(output_dir, exist_ok=True)
generate = True
evaluate = True
filter = True
label_samples = True

# Step 1: load datasets
datasets = load_datasets(datasets)

# Step 2: LM Generation
lm_output_path = os.path.join(output_dir, "generations.jsonl")
if generate:
    num_instances = torch.cuda.device_count()
    batch_size = min(10000, len(datasets) // num_instances + 1)
    llm_actors = [
        RemoteGenerator.options(num_gpus=1, num_cpus=1).remote(**lm_args)
        for _ in range(num_instances)
    ]

    dataset_batches = [
        datasets[i : i + batch_size] for i in range(0, len(datasets), batch_size)
    ]
    futures = ray.util.ActorPool(llm_actors).map_unordered(
        lambda actor, batch_data: actor.generate_batch.remote(batch_data),
        dataset_batches,
    )

    results = []
    for batch_idx, batch_result in enumerate(futures):
        results.extend(batch_result)

    # Step 2.1: filter samples
    filtered_results = []
    for sample in tqdm(results):
        generations = []
        for generation in sample["generation"]:
            # TODO: process for Llama3.2, need to refine
            generation["response"] = generation["response"].replace(
                " I hope it is correct.", ""
            )
            response = generation["response"]
            has_answer = (
                "boxed" in response
                or "he answer is" in response
                or "final answer is" in response
            )
            if has_answer:
                generations.append(generation)
        sample["generation"] = generations
        filtered_results.append(sample)
    save_jsonl(filtered_results, lm_output_path)

# Step 3: Evaluating
eval_output_path = os.path.join(output_dir, "eval_results.jsonl")
if evaluate:
    datasets = load_jsonl(lm_output_path)
    ray.shutdown()
    
    for i in range(0, len(datasets), 10000):
        batch_datasets = datasets[i: i + 10000]
        ray.init()

        futures = []
        for sample in tqdm(batch_datasets, total=len(batch_datasets), desc=f"Extracting Ground Truth"):
            sample["gt_ans"] = parse_ground_truth(sample, data_name=sample["dataset"])[-1]
            for generation in sample["generation"]:
                generation["pred"] = extract_and_strip(generation["response"], data_name=sample["dataset"])
                future = remote_math_equal.remote(generation["pred"], sample["gt_ans"])
                futures.append((future, generation))

        for future, generation in tqdm(futures, total=len(batch_datasets), desc=f"Evaluating samples"):
            try:
                val = ray.get(future, timeout=5)
                generation["correct"] = val
            except ray.exceptions.GetTimeoutError:
                generation["correct"] = False

        ray.shutdown()

    save_jsonl(datasets, eval_output_path)


# Step 3: Split Steps and Extract Valuable samples
filtered_output_path = os.path.join(output_dir, "filtered_sample.jsonl")
if filter:
    datasets = load_jsonl(eval_output_path)
    filtered_datasets = []
    for sample in tqdm(datasets, total=len(datasets), desc="Evaluating samples"):
        accuracy = sum([generation["correct"] for generation in sample["generation"]]) / lm_args["num_sequence"]
        sample["generation_accuracy"] = accuracy
        if accuracy >= 0.75:
            filtered_datasets.append(sample)
    datasets = filtered_datasets
    save_jsonl(datasets, filtered_output_path)


# Step 4: Generate Process Labels
final_output_path = os.path.join(output_dir, "samples_with_labels.jsonl")
if label_samples:
    lm_args["num_sequence"] = num_rollouts
    datasets = load_jsonl(filtered_output_path)
    num_instances = torch.cuda.device_count()

    step_idx = 0
    request_list = init_request_list(datasets)
    while request_list:
        ray.init()
        batch_size = len(request_list) // num_instances + 1
        generators = [
            RemoteStepSearchRequestGenerator.remote(**lm_args)
            for _ in range(num_instances)
        ]
        request_batches = [
            request_list[i: i + batch_size] for i in range(0, len(request_list), batch_size)
        ]

        # Step 4.1: Generate Rollout
        all_outputs = []
        all_pred_list = []
        all_gt_ans_list = []
        futures = ray.util.ActorPool(generators).map(
            lambda actor, batch: actor.generate_rollout.remote(batch), request_batches
        )
        for result in tqdm(futures, total=len(request_batches), desc=f"Generating rollouts for step {step_idx}"):
            outputs, pred_list, gt_ans_list = result
            all_outputs.extend(outputs)
            all_pred_list.extend(pred_list)
            all_gt_ans_list.extend(gt_ans_list)

        # Step 4.2: Evaluate Rollout
        correct_list = []
        # eval_futures = ray.util.ActorPool(evaluators).map(
        #     lambda actor, value: actor.evaluate.remote(value[0], value[1]),
        #     zip(all_pred_list, all_gt_ans_list)
        # )
        # for eval_result in tqdm(eval_futures, total=len(all_outputs), desc=f"Evaluating rollouts for step {step_idx}"):
        #     correct_list.append(eval_result)
        futures = [remote_math_equal.remote(pred, gt_ans) for pred, gt_ans in zip(all_pred_list, all_gt_ans_list)]
        for future in tqdm(futures, total=len(all_pred_list), desc=f"Evaluating rollouts for step {step_idx}"):
            try:
                val = ray.get(future, timeout=5)
                correct_list.append(val)
            except ray.exceptions.GetTimeoutError:
                correct_list.append(False)

        # Step 4.3: Update Request List
        rollout_list = [
            {
                "rollout": rollout,
                "pred": pred,
                "correct": correct,
            }
            for rollout, pred, correct in zip(all_outputs, all_pred_list, correct_list)
        ]
        assert num_rollouts * len(request_list) == len(rollout_list)
        request_list = update_request_list(num_rollouts, request_list, rollout_list)
        step_idx += 1

        ray.shutdown()

        save_jsonl(datasets, final_output_path)

    # ray.shutdown()

