import logging
import os
import sys
from dataclasses import dataclass, field

from datasets import load_dataset

from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from src.train_prm.prm_dataset import ProcessRewardDataset
from src.train_prm.prm_trainer import PRMTrainer


# off wandb
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


@dataclass
class ModelTrainingArguments:
    model_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )


from typing import Optional

@dataclass
class DataTrainingArguments:
    data_path: str = field(
        metadata={
            "help": "The directory for saving the train/dev/test splits and labels."
        }
    )
    input_key: str = field(metadata={"help": "input_key"})
    step_key: str = field(metadata={"help": "step_key"})
    step_label_key: str = field(metadata={"help": "step_label_key"})
    max_length: int = field(metadata={"help": "max length"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


def main():
    parser = HfArgumentParser(
        (ModelTrainingArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize Model and Dataset
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_path,
        num_labels=1,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)

    if training_args.do_train:
        train_dataset = load_dataset(data_args.data_path, split="train")
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.shuffle(seed=42).select(range(max_train_samples))
        train_dataset = ProcessRewardDataset(
            args=data_args, dataset=train_dataset, tokenizer=tokenizer
        )

    if training_args.do_eval:
        eval_dataset = load_dataset(data_args.data_path, split="test")
        eval_dataset = ProcessRewardDataset(
            args=data_args, dataset=eval_dataset, tokenizer=tokenizer
        )

    def data_collator(examples, tokenizer=tokenizer):
        input_ids = []
        input_masks = []
        labels = []
        reward_scores = []
        for input_id, input_mask, label, reward_score in examples:
            input_ids.append(input_id)
            input_masks.append(input_mask)
            labels.append(label)
            reward_scores.append(reward_score)

        pad_token = (
            tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )
        input_ids = pad_sequence(input_ids, padding_value=pad_token, batch_first=True)
        input_masks = pad_sequence(input_masks, padding_value=0, batch_first=True)
        labels = pad_sequence(labels, padding_value=-100, batch_first=True)
        reward_scores = pad_sequence(
            reward_scores, padding_value=-100, batch_first=True
        )
        return {
            "input_ids": input_ids,
            "attention_mask": input_masks,
            "labels": labels,
            "reward_scores": reward_scores,
        }

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        data_size = predictions.shape[0]
        cor_tot, cor_true = 0, 0
        err_tot, err_true = 0, 0
        for idx in range(data_size):
            prediction, label = predictions[idx].squeeze(-1), labels[idx]
            prediction = (prediction[label != -100] > 0).tolist()
            label = label[label != -100].tolist()
            pred_id, label_id = -1, -1
            for step_idx, step_pred in enumerate(prediction):
                if not step_pred:
                    pred_id = step_idx
                    break
            for step_idx, step_label in enumerate(label):
                if not step_label:
                    label_id = step_idx
                    break
            if label_id == -1:
                cor_tot += 1
                cor_true += pred_id == label_id
            else:
                err_tot += 1
                err_true += pred_id == label_id
        cor_acc = cor_true / cor_tot
        err_acc = err_true / err_tot
        f1 = 2 * cor_acc * err_acc / (cor_acc + err_acc)
        return {"correct_accuracy": cor_acc, "error_accuracy": err_acc, "f1": f1}

    trainer = PRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        results = trainer.train()
        trainer.save_model()
        metrics = results.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        logger.info(f"Metrics {metrics}")

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
