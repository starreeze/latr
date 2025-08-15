from dataclasses import asdict
from typing import cast

from transformers.hf_argparser import HfArgumentParser
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from args import DataArgs, ModelArgs, TrainingArgs
from data.loader import get_sft_collator
from data.loader import load as load_datasets
from model.loader import load as load_model
from tools.utils import set_seed


def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainingArgs))  # type: ignore
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args = cast(ModelArgs, model_args)
    data_args = cast(DataArgs, data_args)
    training_args = cast(TrainingArgs, training_args)
    set_seed(training_args.seed)

    model, tokenizer = load_model(model_args)
    train_dataset, test_dataset = load_datasets(data_type="sft", **data_args.__dict__, filter_answer=True)

    training_args_dict = asdict(training_args)
    if training_args_dict["max_steps"] > 0:
        del training_args_dict["num_train_epochs"]
    config = TrainingArguments(**training_args_dict, remove_unused_columns=False)
    collator = get_sft_collator(tokenizer, data_args, train=True)

    trainer = Trainer(
        model=model,
        data_collator=collator,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train(resume_from_checkpoint=bool(training_args.resume_from_checkpoint))
    if config.save_strategy is None or config.save_strategy == "no":
        trainer.save_model()


if __name__ == "__main__":
    main()
