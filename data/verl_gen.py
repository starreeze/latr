# modified from https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py

import os
from dataclasses import dataclass, field
from typing import cast

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers.hf_argparser import HfArgumentParser

template = {
    "base": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: {question}\nAssistant: Let me solve this step by step.\n",
    "inst": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n",
}
math_inst = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{question}\n\nRemember to put your answer on its own line after "Answer:".'


def make_prompt_countdown(sample, template_type):
    target = sample["target"]
    numbers = sample["nums"]
    question = f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    return template[template_type].format(question=question) + "<think>"


def make_prompt_math_w_inst(sample, template_type):
    question = sample["source_prompt"][0]["content"]
    return template[template_type].format(question=question)


def make_prompt_math_raw(sample, template_type, question_key):
    question = math_inst.format(question=sample[question_key])
    return template[template_type].format(question=question)


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
        prompt = make_prompt_math_w_inst(sample, self.template)
        data = {
            "data_source": self.dataset,
            "prompt": prompt,
            "ability": sample.get("ability", "MATH"),
            "reward_model": sample["reward_model"],
            "extra_info": {"split": self.split, "index": idx},
        }
        return data

    def process_aime2024(self, sample, idx):
        prompt = make_prompt_math_raw(sample, self.template, "Problem")
        data = {
            "data_source": self.dataset,
            "prompt": prompt,
            "ability": "MATH",
            "reward_model": {"ground_truth": str(sample["Answer"]), "style": "rule-lighteval/MATH_v2"},
            "extra_info": {"split": self.split, "index": idx},
        }
        return data

    def process_aime2025(self, sample, idx):
        prompt = make_prompt_math_raw(sample, self.template, "question")
        data = {
            "data_source": self.dataset,
            "prompt": prompt,
            "ability": "MATH",
            "reward_model": {"ground_truth": str(sample["answer"]), "style": "rule-lighteval/MATH_v2"},
            "extra_info": {"split": self.split, "index": idx},
        }
        return data

    def process_math500(self, sample, idx):
        prompt = make_prompt_math_raw(sample, self.template, "problem")
        data = {
            "data_source": "HuggingFaceH4/MATH-500",
            "prompt": prompt,
            "ability": "MATH",
            "reward_model": {"ground_truth": str(sample["answer"]), "style": "rule-lighteval/MATH_v2"},
            "extra_info": {"split": self.split, "index": idx},
        }
        return data


def load_raw(name: str, src: str):
    if name == "countdown":
        dataset = load_dataset(src, split="train")
    elif name == "math_dapo":
        dataset = load_dataset(src, "en", split="train")
    elif name == "aime2024":
        dataset = Dataset.from_parquet(os.path.join(src, "2024.parquet"))
    elif name == "aime2025":
        dataset = Dataset.from_json(os.path.join(src, "2025.jsonl"))
    elif name == "math500":
        dataset = load_dataset(src, split="test")
    else:
        raise ValueError(f"Invalid dataset name: {name}")
    return cast(Dataset, dataset)


@dataclass
class Args:
    src: list[str] = field(default_factory=lambda: ["Jiayi-Pan/Countdown-Tasks-3to4"])
    dataset: list[str] = field(default_factory=lambda: ["countdown"])
    output_dir: str = field(default="dataset")
    output_name: str = field(default="")
    train_size: int = field(default=327680)
    test_size: int = field(default=1024)
    template: str = field(default="base")


def main():
    parser = HfArgumentParser([Args])  # type: ignore
    args = cast(Args, parser.parse_args_into_dataclasses()[0])

    trains, tests = [], []

    for dataset, src in zip(args.dataset, args.src):
        raw_dataset = load_raw(dataset, src)

        test_size = args.test_size
        train_size = args.train_size
        if train_size + test_size > len(raw_dataset):
            train_size = len(raw_dataset) - test_size
            if train_size < 0:
                train_size = 0
                test_size = len(raw_dataset)
                print(f"setting test size for {dataset} to max avail {test_size}")
            print(f"setting train size for {dataset} to max avail {train_size}")

        if train_size:
            train_dataset = raw_dataset.select(range(train_size))
            train_processor = SampleProcessor(dataset, args.template, "train")
            train_dataset = train_dataset.map(
                getattr(train_processor, f"process_{dataset}"), with_indices=True
            )
            trains.append(train_dataset)

        if test_size:
            test_processor = SampleProcessor(dataset, args.template, "test")
            test_dataset = raw_dataset.select(range(train_size, train_size + test_size))
            test_dataset = test_dataset.map(getattr(test_processor, f"process_{dataset}"), with_indices=True)
            tests.append(test_dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name if args.output_name else "-".join(args.dataset) + f"-{args.template}"
    local_dir = os.path.join(args.output_dir, output_name)
    if trains:
        train_dataset = concatenate_datasets(trains)
        train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    if tests:
        test_dataset = concatenate_datasets(tests)
        test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))


if __name__ == "__main__":
    main()
