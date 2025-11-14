# modified from https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers.hf_argparser import HfArgumentParser

from tools.utils import snake_to_camel

template = {
    "base": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: {question}\nAssistant: Let me solve this step by step.\n",
    "inst-qwen": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n",
    "inst-ds": "<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n",
    "inst-phi": "<|system|>You are a helpful assistant.<|end|><|user|>{question}<|end|><|assistant|>",
}

useful_cloumns = ["prompt", "reward_model", "data_source"]


class DataProcessor(ABC):
    test_only: bool = False

    def __init__(self, dataset_name, template, src, train_size, test_size):
        self.name = dataset_name
        self.template = template
        self.src = src

        dataset = self.load_raw()

        if self.test_only:
            train_size = 0
            test_size = len(dataset)
        elif train_size + test_size > len(dataset):
            train_size = len(dataset) - test_size
            if train_size < 0:
                train_size = 0
                test_size = len(dataset)

        assert train_size >= 0 and test_size >= 0
        assert train_size + test_size <= len(dataset)

        print(f"setting test size for {dataset_name} to {test_size}")
        print(f"setting train size for {dataset_name} to {train_size}")

        if train_size:
            train_dataset = dataset.select(range(train_size))
            self.train_dataset = self.to_verl(train_dataset)
        else:
            self.train_dataset = None

        if test_size:
            test_dataset = dataset.select(range(train_size, train_size + test_size))
            self.test_dataset = self.to_verl(test_dataset)
        else:
            self.test_dataset = None

    @abstractmethod
    def load_raw(self) -> Dataset:
        pass

    @abstractmethod
    def process_sample(self, sample) -> dict:
        pass

    def to_verl(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(self.process_sample)
        dataset = dataset.remove_columns([c for c in dataset.column_names if c not in useful_cloumns])
        return dataset


class CountdownProcessor(DataProcessor):
    @staticmethod
    def make_prompt(sample, template_type):
        target = sample["target"]
        numbers = sample["nums"]
        question = f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
        return template[template_type].format(question=question) + "<think>"

    def load_raw(self):
        return load_dataset(self.src, split="train")

    def process_sample(self, sample):
        prompt = self.make_prompt(sample, self.template)
        solution = {"target": sample["target"], "numbers": sample["nums"]}
        data = {
            "data_source": self.name,
            "prompt": prompt,
            "reward_model": {"style": "rule", "ground_truth": solution},
        }
        return data


class MathProcessor(DataProcessor):
    question_key: str
    answer_key: str

    @staticmethod
    def make_prompt(question, template_type):
        math_inst = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{question}\n\nRemember to put your answer on its own line after "Answer:".'
        question = math_inst.format(question=question)
        return template[template_type].format(question=question)

    def process_sample(self, sample):
        return {
            "data_source": self.name,
            "prompt": self.make_prompt(sample[self.question_key], self.template),
            "reward_model": {"style": "rule", "ground_truth": str(sample[self.answer_key])},
        }


class Gsm8kProcessor(MathProcessor):
    test_only = True

    def load_raw(self):
        return load_dataset(self.src, "main", split="test")

    def process_sample(self, sample):
        answer = sample["answer"].rsplit("####", 1)[-1].strip()
        return {
            "data_source": self.name,
            "prompt": self.make_prompt(sample["question"], self.template),
            "reward_model": {"style": "rule", "ground_truth": answer},
        }


class MathDapoProcessor(MathProcessor):
    def load_raw(self):
        return load_dataset(self.src, "en", split="train")

    def process_sample(self, sample):
        return {
            "data_source": self.name,
            "prompt": self.make_prompt(sample["prompt"], self.template),
            "reward_model": {"style": "rule", "ground_truth": str(sample["reward_model"]["ground_truth"])},
        }


class Aime2024Processor(MathProcessor):
    question_key = "Problem"
    answer_key = "Answer"
    test_only = True

    def load_raw(self):
        return Dataset.from_parquet(os.path.join(self.src, "2024.parquet"))


class Aime2025Processor(MathProcessor):
    question_key = "question"
    answer_key = "answer"
    test_only = True

    def load_raw(self):
        return Dataset.from_json(os.path.join(self.src, "2025.jsonl"))


class Math500Processor(MathProcessor):
    question_key = "question"
    answer_key = "answer"
    test_only = True

    def load_raw(self):
        return Dataset.from_json(os.path.join(self.src, f"{self.name}.jsonl"))


OlympiadProcessor = Math500Processor


class Amc23Processor(MathProcessor):
    question_key = "question"
    answer_key = "answer"
    test_only = True

    def load_raw(self):
        return load_dataset(self.src, split="test")


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
        dp_cls = globals()[f"{snake_to_camel(dataset)}Processor"]
        processor: DataProcessor = dp_cls(dataset, args.template, src, args.train_size, args.test_size)
        if processor.train_dataset:
            trains.append(processor.train_dataset)
        if processor.test_dataset:
            tests.append(processor.test_dataset)

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
