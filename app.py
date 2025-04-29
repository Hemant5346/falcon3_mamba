import os
import json
import re
from tqdm import tqdm
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# === Setup ===
qwen_model_id = "Qwen/Qwen2.5-Math-7B-Instruct"
prm_model_id = "Qwen/Qwen2.5-Math-PRM-7B"
dataset_path = "./eval_data/math/test.jsonl"  # Adjust path if needed

system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
max_new_tokens = 512

# === Detect device ===
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"[INFO] Using device: {device}")

# === Load Qwen LLM ===
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    device_map={"": device},
    torch_dtype=torch_dtype,
    trust_remote_code=True,
).eval()

# === Load PRM model ===
prm_tokenizer = AutoTokenizer.from_pretrained(prm_model_id, trust_remote_code=True)
prm_model = AutoModel.from_pretrained(
    prm_model_id,
    device_map={"": device},
    torch_dtype=torch_dtype,
    trust_remote_code=True,
).eval()

# === Utility Functions ===

def generate_response(question: str) -> List[str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tokenizer(text, return_tensors="pt").to(device)
    outputs = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)
    full_text = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    steps = re.split(r"<extra_0>", full_text)
    return [s.strip() for s in steps if s.strip()]

def extract_final_answer(response: str) -> str:
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    return match.group(1).strip() if match else None

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        scores = positive_probs.cpu().tolist()
        all_scores_res.append(scores)
    return all_scores_res

def score_response_with_prm(query: str, steps: List[str]) -> float:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}
    ]
    text = prm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = prm_tokenizer(text, return_tensors="pt").to(device)
    outputs = prm_model(input_ids=inputs.input_ids)
    sep_token_id = prm_tokenizer.encode("<extra_0>")[0]
    token_mask = (inputs.input_ids == sep_token_id)
    rewards = make_step_rewards(outputs[0], token_mask)
    return sum(rewards[0]) / len(rewards[0]) if rewards and rewards[0] else 0.0

# === Run Evaluation ===
print(f"\n🔍 Evaluating math benchmark...")

with open(dataset_path) as f:
    samples = [json.loads(line) for line in f]

total, correct = 0, 0
wrong_answers = []

for sample in tqdm(samples, desc="Processing math"):
    question = sample.get("question") or sample.get("problem")
    gt = sample["answer"]
    try:
        steps = generate_response(question)
        final_answer = extract_final_answer(steps[-1]) if steps else None
        score = score_response_with_prm(question, steps)
    except Exception as e:
        print(f"⚠️ Error processing: {e}")
        continue

    is_correct = final_answer is not None and final_answer.strip() == gt.strip()
    total += 1
    correct += is_correct

    if not is_correct:
        wrong_answers.append({
            "question": question,
            "ground_truth": gt,
            "predicted": final_answer,
            "steps": steps,
            "reward_score": score
        })

acc = correct / total if total > 0 else 0
print(f"\n✅ Accuracy on math: {correct}/{total} = {acc:.2%}")

# Save wrong answers
os.makedirs("results", exist_ok=True)
with open("results/math_wrong.jsonl", "w") as f:
    for item in wrong_answers:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
