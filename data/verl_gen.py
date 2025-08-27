# modified from https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py

import argparse
import os
from typing import cast

from datasets import Dataset, load_dataset


def make_prompt_countdown(sample, template_type):
    target = sample["target"]
    numbers = sample["nums"]
    if template_type == "base":
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.\n<think>"""
    elif template_type == "inst":
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


def make_prompt_math(sample, template_type):
    question = sample["source_prompt"][0]["content"]
    if template_type == "base":
        prefix = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: {question}\nAssistant: Let me solve this step by step.\n"
    elif template_type == "inst":
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n"""
    return prefix


class SampleProcessor:
    def __init__(self, dataset, template, split):
        self.dataset = dataset
        self.template = template
        self.split = split

    def process_countdown(self, sample, idx):
        prompt = make_prompt_countdown(sample, self.template)
        solution = {"target": sample["target"], "numbers": sample["nums"]}
        data = {
            "data_source": self.dataset,
            "prompt": prompt,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": self.split, "index": idx},
        }
        return data

    def process_math_dapo(self, sample, idx):
        prompt = make_prompt_math(sample, self.template)
        data = {
            "data_source": self.dataset,
            "prompt": prompt,
            "ability": sample.get("ability", "MATH"),
            "reward_model": sample["reward_model"],
            "extra_info": {"split": self.split, "index": idx},
        }
        return data


def load_raw(name, src):
    if name == "countdown":
        dataset = load_dataset(src, split="train")
    elif name == "math_dapo":
        dataset = load_dataset(src, "en", split="train")
    else:
        raise ValueError(f"Invalid dataset name: {name}")
    return cast(Dataset, dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="Jiayi-Pan/Countdown-Tasks-3to4")
    parser.add_argument("--dataset", default="countdown")  # math_dapo
    parser.add_argument("--output_dir", default="dataset")
    parser.add_argument("--train_size", type=int, default=327680)
    parser.add_argument("--test_size", type=int, default=1024)
    parser.add_argument("--template", type=str, default="base")
    args = parser.parse_args()

    raw_dataset = load_raw(args.dataset, args.src)

    if args.train_size + args.test_size > len(raw_dataset):
        args.train_size = len(raw_dataset) - args.test_size
        print(f"setting train size to max avail {args.train_size}")
    train_dataset = raw_dataset.select(range(args.train_size))
    test_dataset = raw_dataset.select(range(args.train_size, args.train_size + args.test_size))

    train_processor = SampleProcessor(args.dataset, args.template, "train")
    test_processor = SampleProcessor(args.dataset, args.template, "test")
    train_dataset = train_dataset.map(getattr(train_processor, f"process_{args.dataset}"), with_indices=True)
    test_dataset = test_dataset.map(getattr(test_processor, f"process_{args.dataset}"), with_indices=True)

    local_dir = os.path.join(args.output_dir, args.dataset) + f"-{args.template}"
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))


if __name__ == "__main__":
    main()
