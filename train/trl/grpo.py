import os
from dataclasses import asdict
from typing import cast

import torch._dynamo
from transformers.hf_argparser import HfArgumentParser

from args import DataArgs, GRPOArgs, ModelArgs, RewardArgs
from data.loader import load as load_datasets
from model.generation import KeyTokenGenMixin
from model.loader import load as load_model
from reward import datasets, format, latent
from tools.utils import init_dataclass_from_dict, set_seed
from train.trl.key_token import KeyTokenGRPOConfig, KeyTokenGRPOTrainer
from train.trl.pretty import PrettyGRPOConfig, PrettyGRPOTrainer


def get_reward_funcs(data_args: DataArgs, reward_args: RewardArgs, is_reasoning_model: bool):
    assert data_args.cot_type != "none", "GRPO does not support none cot type"
    correctness_fn = getattr(datasets, f"{data_args.dataset_name.capitalize()}CorrectnessReward")(
        reward_args, data_args, is_reasoning_model
    )
    reward_funcs = [
        correctness_fn,
        getattr(format, f"{data_args.cot_type.capitalize()}LooseFormatReward")(
            reward_args, data_args, is_reasoning_model
        ),
        getattr(format, f"{data_args.cot_type.capitalize()}StrictFormatReward")(
            reward_args, data_args, is_reasoning_model
        ),
        getattr(datasets, f"{data_args.dataset_name.capitalize()}AnswerFormatReward")(
            reward_args, data_args, is_reasoning_model
        ),
    ]
    if reward_args.answer_len_reward > 0:
        reward_funcs.append(latent.AnswerLenReward(reward_args, data_args, is_reasoning_model))
    if reward_args.answer_no_think_reward > 0:
        reward_funcs.append(latent.AnswerNoThinkReward(reward_args, data_args, is_reasoning_model))
    if reward_args.response_think_reward > 0:
        reward_funcs.append(latent.ResponseThinkReward(reward_args, data_args, is_reasoning_model))
    return reward_funcs


def get_trainer(model, tokenizer, grpo_args_dict, train_dataset, test_dataset, reward_funcs):
    if isinstance(model, KeyTokenGenMixin):
        config = init_dataclass_from_dict(KeyTokenGRPOConfig, grpo_args_dict)
        trainer = KeyTokenGRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
    else:
        config = init_dataclass_from_dict(PrettyGRPOConfig, grpo_args_dict)
        if not config.use_vllm:
            # since use_cache is set to False to fix generation padding issue in model/loader.py,
            # the performance of vanilla model is terrible.
            raise ValueError("Consider using vLLM for better performance when training vanilla model.")
        trainer = PrettyGRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
    return config, trainer


def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, GRPOArgs, RewardArgs))  # type: ignore
    model_args, data_args, grpo_args, reward_args = parser.parse_args_into_dataclasses()
    model_args = cast(ModelArgs, model_args)
    data_args = cast(DataArgs, data_args)
    grpo_args = cast(GRPOArgs, grpo_args)
    reward_args = cast(RewardArgs, reward_args)
    set_seed(grpo_args.seed)
    torch._dynamo.config.suppress_errors = True  # type: ignore

    model, tokenizer = load_model(model_args)
    train_dataset, test_dataset = load_datasets(data_type="rl", **data_args.__dict__)

    grpo_args_dict = asdict(grpo_args)
    if grpo_args_dict["max_steps"] > 0:
        del grpo_args_dict["num_train_epochs"]

    name = os.path.basename(model_args.model).lower()
    reasoning_names = ["qwen3", "deepseek"]
    is_reasoning_model = "base" not in name and any(r in name for r in reasoning_names)
    reward_funcs = get_reward_funcs(data_args, reward_args, is_reasoning_model)
    config, trainer = get_trainer(model, tokenizer, grpo_args_dict, train_dataset, test_dataset, reward_funcs)

    resume = bool(grpo_args.resume_from_checkpoint)
    if resume:
        print("Resuming from checkpoint ...")
    trainer.train(resume_from_checkpoint=resume)
    if config.save_strategy is None or config.save_strategy == "no":
        trainer.save_model()


if __name__ == "__main__":
    main()
