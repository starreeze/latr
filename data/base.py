import os
import warnings
from typing import cast

from datasets import Dataset, Value, concatenate_datasets, load_dataset

from data.prompts import PromptFormatter


def select_data(data: Dataset, split: str, n_train_samples: int, n_eval_samples: int) -> Dataset:
    if split == "train":
        if n_train_samples > len(data):
            warnings.warn(f"{n_train_samples=} > {len(data)=}; setting n_train_samples to {len(data)}")
        else:
            data = data.select(range(n_train_samples))
    else:
        if n_eval_samples > len(data):
            warnings.warn(f"{n_eval_samples=} > {len(data)=}; setting n_eval_samples to {len(data)}")
        else:
            data = data.select(range(n_eval_samples))
    return data


def load_raw_gsm8k(data_path: str, split: str) -> Dataset:
    data = cast(Dataset, load_dataset(data_path, "main", split=split))
    return data


def load_raw_countdown(data_path: str, split: str) -> Dataset:
    file_name = "train.parquet" if split == "train" else "test.parquet"
    data = cast(Dataset, Dataset.from_parquet(os.path.join(data_path, file_name)))
    columns_to_remove = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    data = data.remove_columns([col for col in columns_to_remove if col in data.column_names])
    return data


def load_aime_train(data_path: str) -> Dataset:
    data = cast(Dataset, Dataset.from_csv(os.path.join(data_path, "1983-2023.csv")))
    columns_to_remove = ["ID", "Year", "Problem Number", "Part"]
    data = data.remove_columns([col for col in columns_to_remove if col in data.column_names])
    data = data.rename_column("Question", "question").rename_column("Answer", "answer")
    return data


def load_aime_test(data_path: str) -> Dataset:
    data_24 = cast(Dataset, Dataset.from_parquet(os.path.join(data_path, "2024.parquet")))
    data_24 = data_24.remove_columns(["ID", "Solution"])
    data_24 = data_24.rename_column("Problem", "question").rename_column("Answer", "answer")
    data_24 = data_24.cast_column("answer", Value("string"))
    data_25 = cast(Dataset, Dataset.from_json(os.path.join(data_path, "2025.jsonl")))
    data = concatenate_datasets([data_24, data_25])
    return data


def load_raw_aime(data_path: str, split: str) -> Dataset:
    if split == "train":
        data = load_aime_train(data_path)
    else:
        data = load_aime_test(data_path)
    return data


def load_raw_dapomath(data_path: str, split: str) -> Dataset:
    """
    To download the dataset, run:
    huggingface-cli download open-r1/DAPO-Math-17k-Processed --repo-type dataset --local-dir dataset/dapomath
    """
    assert split == "train", (
        "dapomath only has train split. "
        "To align with common practice, use aime for eval by specifying `--eval_dataset aime`."
    )
    data = cast(Dataset, load_dataset(data_path, "en", split="train"))
    data = data.remove_columns(["data_source", "source_prompt", "ability", "reward_model", "extra_info"])
    return data


def load_raw_math500(data_path: str, split: str) -> Dataset:
    """
    To download the dataset, run:
    huggingface-cli download HuggingFaceH4/MATH-500 --repo-type dataset --local-dir dataset/math500
    """
    assert split == "test", (
        "math500 only has test split. "
        "Consider using dapomath for training by specifying `--dataset_name dapomath`."
    )
    data = cast(Dataset, load_dataset(data_path, split=split))
    data = data.remove_columns(["subject", "level", "unique_id"])
    return data


class DataBase:
    "The base class for all data loading classes."

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        split: str,
        n_train_samples: int,
        n_eval_samples: int,
        cot_type: str,
        system_prompt_type: str,
        template: str,
        filter_answer: bool = False,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        sys_prompt = PromptFormatter.get_system_prompt(system_prompt_type)
        self.system_prompt_message = [{"role": "system", "content": sys_prompt}] if sys_prompt else []
        self.prompt_formatter = PromptFormatter(
            dataset_name, cot_type, kwargs["plain_cot_start_marker"], kwargs["plain_cot_end_marker"]
        )
        self.template = template

        if filter_answer and (split != "train" or dataset_name != "countdown"):
            warnings.warn("filter_answer is only supported for countdown train split")
            filter_answer = False

        data_path = os.path.join(data_dir, dataset_name)
        data = globals()[f"load_raw_{dataset_name}"](data_path, split)
        if filter_answer:
            original_len = len(data)
            data = data.filter(lambda x: x["answer"] != "")
            print(f"filtered {original_len} -> {len(data)} samples")
        self.data = select_data(data, split, n_train_samples, n_eval_samples)

    def load(self) -> Dataset:
        return self.data.map(getattr(self, f"process_{self.dataset_name}_sample"))
