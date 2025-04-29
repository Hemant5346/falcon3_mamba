import time

import torch
from torch import sigmoid
from transformers import AutoModelForTokenClassification, AutoTokenizer
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None


from src.eval_utils.prompt_template import PromptTemplate


class Generator:
    def __init__(
        self,
        model_path,
        prompt_template,
        num_sequence=1,
        max_tokens=2048,
        max_model_len=4096,
        temperature=0.0,
        top_p=1.0,
        stop_tokens=None,
        stop_token_ids=None,
        tensor_parallel_size=1,
    ):
        seed = int(time.time() * 1e6) % int(1e9)
        self.prompt_func = PromptTemplate.load_from_id_or_path(
            prompt_template=prompt_template
        )
        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_prefix_caching=False,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            seed=seed,
            enforce_eager=True,
        )
        self.n_sampling = num_sequence
        self.sampling_params = SamplingParams(
            n=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_tokens,
            stop_token_ids=stop_token_ids,
        )

    def generate(self, prompts):
        return self.llm.generate(prompts, self.sampling_params)

    def generate_batch(self, samples) -> dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        prompts = [
            self.prompt_func.make_full_prompt(sample["problem"])
            for sample in samples
            for _ in range(self.n_sampling)
        ]
        outputs = self.llm.generate(prompts, self.sampling_params)
        for idx, sample in enumerate(samples):
            # sample["prompt"] = prompt
            sample.update(
                {
                    # "prompt": prompt,
                    # "agent": self.model_path,
                    "generation": [
                        {"response": o.outputs[0].text}
                        for o in outputs[
                            idx * self.n_sampling : (idx + 1) * self.n_sampling
                        ]
                    ],
                }
            )
        return samples


class PRMPredictor:
    def __init__(self, model_path):
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def process_sample(self, sample):
        question = sample["problem"] + "\n\n"
        steps = sample["steps"]
        step_token_id = self.tokenizer.encode("\n\n")[-1]

        input_ids = self.tokenizer.encode(question, add_special_tokens=False)
        attention_mask = [1] * len(input_ids)
        reward_flags = [0] * len(input_ids)
        for step in steps:
            step_tokens = self.tokenizer.encode(step, add_special_tokens=False)
            input_ids.extend(step_tokens + [step_token_id])
            attention_mask.extend([1] * (len(step_tokens) + 1))
            reward_flags.extend([0] * len(step_tokens) + [1])

        input_ids = torch.tensor(input_ids, dtype=torch.int)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int)
        reward_flags = torch.tensor(reward_flags, dtype=torch.bool)

        return (input_ids, attention_mask, reward_flags)
    
    def _get_step_rewards(self, input_ids, attention_mask, reward_flags):
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.model.device)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits.detach().cpu().view(-1)
        step_logits = logits[reward_flags]
        return sigmoid(step_logits).tolist()

    def score(self, sample):
        if "generation" in sample:
            for generation in sample["generation"]:
                single_response_sample = {
                    "problem": sample["problem"],
                    "steps": [step.strip() for step in generation["response"].split("\n\n")]
                    # "steps": split_steps(generation["response"]),
                }
                input_ids, attention_mask, reward_flags = self.process_sample(single_response_sample)
                generation["step_rewards"] = self._get_step_rewards(input_ids, attention_mask, reward_flags)
        else:
            input_ids, attention_mask, reward_flags = self.process_sample(sample)
            sample["step_rewards"] = self._get_step_rewards(input_ids, attention_mask, reward_flags)
        
        return sample


def split_steps(response):
    steps = []
    now_step = ""
    all_steps = response.split("\n\n")
    for step_str in all_steps:
        if step_str.startswith("\\[") or step_str.startswith("   "):
            if now_step.strip():
                now_step += "\n"
            now_step += step_str
        else:
            if now_step.strip():
                steps.append(now_step)
            now_step = step_str
    if now_step.strip():
        steps.append(now_step)
    return steps
