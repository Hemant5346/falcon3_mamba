import torch.nn as nn

from transformers import Trainer


class PRMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = nn.BCELoss()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        reward_logits = outputs.logits
        labels = inputs["labels"]
        reward_scores = inputs["reward_scores"]

        reward_positions = (labels != -100).view(-1)
        reward_logits = reward_logits.view(-1)[reward_positions]
        reward_scores = reward_scores.view(-1)[reward_positions]

        loss = self.loss_func(reward_logits.sigmoid(), reward_scores)

        return (loss, outputs) if return_outputs else loss
