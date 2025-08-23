import glob
import os
import random
from typing import Any, cast

from datasets import Dataset, concatenate_datasets
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser

from args import DataArgs, GenerationArgs, ModelArgs, TrainingArgs
from data import eval as answer_processor
from data.loader import get_sft_collator
from data.loader import load as load_datasets
from data.utils import equal_level
from model.generation import KeyTokenGenMixin
from model.loader import load as load_model
from tools.multiprocess import MultiGPUManager, MultiGPUWorker
from tools.utils import to_device


class DataWorker(MultiGPUWorker):
    """Worker class for evaluation across multiple GPUs."""

    def process_chunk(self, dataset_chunk: Dataset, gpu_id: int, device: str, **kwargs):
        """Process a chunk of dataset for evaluation."""
        model_args: ModelArgs = kwargs["model_args"]
        data_args: DataArgs = kwargs["data_args"]
        generation_args: GenerationArgs = kwargs["generation_args"]

        print(f"Worker {gpu_id}: Loading model to {device}")
        model, tokenizer = load_model(model_args)
        model.to(device)  # type: ignore
        model.eval()
        assert isinstance(model, KeyTokenGenMixin)

        # Set up data processing
        data_args.align_label_with_tf_mask = False
        collator = get_sft_collator(tokenizer, data_args, keep_original_data=True)
        loader = DataLoader(dataset_chunk, batch_size=data_args.generate_batch_size, collate_fn=collator)  # type: ignore
        answer_extractor = answer_processor.AnswerExtractor(data_args)

        print(f"Worker {gpu_id}: Evaluating {len(dataset_chunk)} samples")
        all_pairs: list[list[tuple[Tensor, Tensor, Tensor]]] = [[], [], []]

        for batch in tqdm(
            loader, total=len(loader), desc=f"eval {data_args.eval_dataset} GPU {gpu_id}", position=gpu_id
        ):
            inputs: dict[str, Any] = to_device(batch, device)  # type: ignore
            question_raw = inputs["question_ids"].repeat_interleave(
                generation_args.num_return_sequences, dim=0
            )
            attention_mask = inputs["attention_mask"].repeat_interleave(
                generation_args.num_return_sequences, dim=0
            )

            results = model.generate(question_raw, tokenizer, attention_mask, **generation_args.__dict__)
            branches = results.branch_info
            assert branches is not None
            pred_raw = results.sequences[: len(branches)]

            pred_raw_strs = tokenizer.batch_decode(pred_raw, skip_special_tokens=True)
            prompt_len = question_raw.shape[1]
            answers = []

            for question_id, pred_raw_str in zip(question_raw, pred_raw_strs):
                question_str = tokenizer.decode(question_id, skip_special_tokens=True)
                assert pred_raw_str.startswith(question_str)
                pred_raw_str = pred_raw_str[len(question_str) :]
                answers.append(answer_extractor(pred_raw_str))

            sample_pairs = [[], [], []]
            for i, branch in enumerate(branches):
                if branch.parent is None:
                    continue
                if not answers[i] or not answers[branch.parent]:
                    continue  # those without a stop should not be counted
                label = equal_level(answers[i], answers[branch.parent])
                start = prompt_len + branch.birth_step
                prefix = pred_raw[i, prompt_len:start]
                if data_args.diverge_max_prefix_len is not None:
                    prefix = prefix[-data_args.diverge_max_prefix_len :]
                prefix = tokenizer.decode(prefix, skip_special_tokens=True).strip()
                for k in data_args.diverge_seqlen_k:
                    a = pred_raw[i, start : start + k]
                    b = pred_raw[branch.parent, start : start + k]
                    if random.random() < 0.5:
                        a, b = b, a
                    a = tokenizer.decode(a, skip_special_tokens=True).strip()
                    b = tokenizer.decode(b, skip_special_tokens=True).strip()
                    sample_pairs[label].append((prefix, a, b))

            if data_args.diverge_balance_label:
                # balance the label by subsampling for each sample to eliminate bias across samples
                min_len = min(len(sample_pairs[0]), len(sample_pairs[2]))
                for i in range(3):
                    sample_pairs[i] = random.sample(sample_pairs[i], min(len(sample_pairs[i]), min_len))

            for all_p, sample_p in zip(all_pairs, sample_pairs):
                all_p.extend(sample_p)

        print(
            f"Worker {gpu_id}: Finished, label 0: {len(all_pairs[0])}, "
            f"label 1: {len(all_pairs[1])}, label 2: {len(all_pairs[2])}"
        )
        return all_pairs

    def combine_results(self, results: list[list[list[tuple[Tensor, Tensor, Tensor]]]]) -> list[dict]:
        """Combine results from multiple workers."""
        all_data = []
        for label2data in results:
            for label, data in enumerate(label2data):
                for prefix, a, b in data:
                    all_data.append({"prefix": prefix, "a": a, "b": b, "label": label / 2})
        return all_data


def construct_dataset():
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainingArgs, GenerationArgs))  # type: ignore
    model_args, data_args, train_args, generation_args = parser.parse_args_into_dataclasses()
    model_args = cast(ModelArgs, model_args)
    data_args = cast(DataArgs, data_args)
    train_args = cast(TrainingArgs, train_args)
    generation_args = cast(GenerationArgs, generation_args)

    train_dataset, _ = load_datasets(data_type="sft", **data_args.__dict__)
    if data_args.end_pos < len(train_dataset) or data_args.start_pos > 0:
        start_pos = max(0, data_args.start_pos)
        end_pos = min(len(train_dataset), data_args.end_pos)
        print(f"Selecting {start_pos} to {end_pos} from {len(train_dataset)} samples")
        train_dataset = train_dataset.select(range(start_pos, end_pos))
    else:
        start_pos = 0

    n_gpus = data_args.n_gpus
    print(f"Using {n_gpus} GPUs for divergence dataset construction.")

    manager = MultiGPUManager(DataWorker)
    results = manager.run(
        train_dataset, n_gpus, model_args=model_args, data_args=data_args, generation_args=generation_args
    )
    dataset = Dataset.from_list(results)
    model_name = os.path.basename(model_args.model.rstrip("/"))
    dataset.to_parquet(f"{data_args.data_dir}/diverge/{model_name}_{start_pos}_{end_pos}.parquet")


def load_dataset(path: str, test_ratio: float = 0.05):
    paths = glob.glob(os.path.join(path, "*.parquet"))
    datasets: dict[str, list[Dataset]] = {"qwen2.5": [], "qwen3": []}
    for path in paths:
        dataset = cast(Dataset, Dataset.from_parquet(path))
        if "Qwen2.5" in path:
            datasets["qwen2.5"].append(dataset)
        elif "Qwen3" in path:
            datasets["qwen3"].append(dataset)
    data_2_5 = concatenate_datasets(datasets["qwen2.5"])
    data_3 = concatenate_datasets(datasets["qwen3"])
    min_len = min(len(data_2_5), len(data_3))
    data_2_5 = data_2_5.select(range(min_len))
    data_3 = data_3.select(range(min_len))
    dataset = concatenate_datasets([data_2_5, data_3])
    dataset = dataset.filter(lambda x: x["label"] == 0 or x["label"] == 1)
    dataset = dataset.train_test_split(test_size=test_ratio, shuffle=True)
    return dataset["train"], dataset["test"]


if __name__ == "__main__":
    construct_dataset()
