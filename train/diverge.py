from dataclasses import dataclass
from typing import cast

import torch
from transformers import AutoTokenizer
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from data.diverge import load_dataset
from model.diverge import DivCollator, DivJudge
from tools.utils import set_seed


@dataclass
class DivergenceArgs:
    load_path: str = "Qwen/Qwen3-Embedding-0.6B"
    save_path: str = "outputs/diverge"
    data_path: str = "dataset/diverge"
    max_prefix_len: int = 100


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    return {
        "accuracy": (torch.abs(preds - labels) < 0.25).float().mean().item(),
        "mae": torch.abs(preds - labels).mean().item(),
        "preds_mean": preds.mean().item(),
        "labels_mean": labels.mean().item(),  # type: ignore
    }


class DivergenceTrainer(Trainer):
    """Custom trainer that logs metrics at every training step."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss and log metrics for each training step."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss

        metrics = compute_metrics(outputs.scores, labels)
        metrics["loss"] = float(loss.item())
        self.log({key: value for key, value in metrics.items()})

        return (loss, outputs) if return_outputs else loss


def main():
    parser = HfArgumentParser((TrainingArguments, DivergenceArgs))  # type: ignore
    training_args, divergence_args = parser.parse_args_into_dataclasses()
    training_args = cast(TrainingArguments, training_args)
    divergence_args = cast(DivergenceArgs, divergence_args)
    training_args.remove_unused_columns = False
    training_args.output_dir = divergence_args.save_path
    training_args.save_strategy = "no"
    set_seed(training_args.seed)

    model = DivJudge(divergence_args.load_path, max_prefix_len=divergence_args.max_prefix_len)
    tokenizer = AutoTokenizer.from_pretrained(divergence_args.load_path)
    train_dataset, test_dataset = load_dataset(divergence_args.data_path)
    collator = DivCollator(tokenizer)

    trainer = DivergenceTrainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train(resume_from_checkpoint=bool(training_args.resume_from_checkpoint))
    if training_args.save_strategy is None or training_args.save_strategy == "no":
        model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
