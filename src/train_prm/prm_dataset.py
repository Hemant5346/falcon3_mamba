import torch
from torch.utils.data import Dataset


class ProcessRewardDataset(Dataset):
    """
    Dataset for process reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(self, args, dataset, tokenizer) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = args.max_length

        # chat_template
        self.input_key = getattr(self.args, "input_key", None)
        self.step_key = getattr(self.args, "step_key", None)
        self.step_label_key = getattr(self.args, "step_label_key", None)

        # Store the processed data in class attributes
        self.question_list = dataset[self.input_key]
        self.steps_list = dataset[self.step_key]
        self.step_scores_list = dataset[self.step_label_key]

    def __len__(self):
        length = len(self.question_list)
        return length

    def __getitem__(self, idx):
        question = self.question_list[idx] + "\n\n"
        steps = self.steps_list[idx]
        step_token_id = self.tokenizer.encode("\n\n")[-1]
        steps_score = self.step_scores_list[idx]

        input_ids = self.tokenizer.encode(question, add_special_tokens=False)
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(input_ids)
        reward_scores = [-100] * len(input_ids)
        for step, score in zip(steps, steps_score):
            step_tokens = self.tokenizer.encode(step, add_special_tokens=False)
            input_ids.extend(step_tokens + [step_token_id])
            attention_mask.extend([1] * (len(step_tokens) + 1))
            labels.extend([-100] * len(step_tokens) + [1 if score >= 0.5 else 0])
            reward_scores.extend([-100] * len(step_tokens) + [score])

        input_ids = torch.tensor(input_ids[: self.max_length], dtype=torch.int)
        attention_mask = torch.tensor(
            attention_mask[: self.max_length], dtype=torch.int
        )
        labels = torch.tensor(labels[: self.max_length], dtype=torch.int)
        reward_scores = torch.tensor(
            reward_scores[: self.max_length], dtype=torch.float
        )

        return (input_ids, attention_mask, labels, reward_scores)
