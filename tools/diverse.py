from __future__ import annotations

import csv
import json
import os
import re
import shutil
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from glob import glob
from itertools import combinations
from typing import Any, cast

from datasets import Dataset
from natsort import natsorted
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.hf_argparser import HfArgumentParser

from data.utils import eval_countdown_equation, validate_countdown_equation
from model.generation import KeyTokenGenConfig, KtModules, generate


def make_prompt_countdown(sample, template="base"):
    target = sample["target"]
    numbers = sample["nums"]
    if template == "base":
        prefix = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.\n<think>"
    elif template == "inst":
        prefix = f"You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n<|im_start|>user\nUsing the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"
    return prefix


def pairwise_similarity(
    inputs: list[str],
    bleu_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    rouge_types: list[str] = ["rougeL"],
    use_stemmer: bool = False,
) -> tuple[float, float]:
    """
    Calculates average pairwise self-BLEU and self-ROUGE metrics for a list of generated texts.
    Returns (0.0, 0.0) if less than 2 inputs are provided.
    Higher scores mean less diversity (more similarity).
    """
    if len(inputs) < 2:
        return (0.0, 0.0)

    total_bleu = 0.0
    total_rouge_f1 = 0.0
    pair_count = 0

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=use_stemmer)
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    for sentence1, sentence2 in combinations(inputs, 2):
        # BLEU score calculation
        reference = [sentence1.split()]
        candidate = sentence2.split()
        try:
            bleu_score = sentence_bleu(
                reference, candidate, weights=bleu_weights, smoothing_function=SmoothingFunction().method1
            )
        except (ValueError, ZeroDivisionError) as e:
            warnings.warn(f"BLEU calculation failed for pair: {e}")
            bleu_score = 0.0

        # ROUGE score calculation
        try:
            rouge_scores = scorer.score(sentence1, sentence2)
            rouge_f1_avg = sum(score.fmeasure for score in rouge_scores.values()) / len(rouge_scores)
        except Exception as e:
            warnings.warn(f"ROUGE calculation failed for pair: {e}")
            rouge_f1_avg = 0.0

        total_bleu += bleu_score  # type: ignore
        total_rouge_f1 += rouge_f1_avg
        pair_count += 1

    avg_bleu = total_bleu / pair_count if pair_count > 0 else 0.0
    avg_rouge_f1 = total_rouge_f1 / pair_count if pair_count > 0 else 0.0

    # Return 1 - score to represent diversity (0 = identical, 1 = completely different)
    return (1 - avg_bleu, 1 - avg_rouge_f1)


@dataclass
class EvalArgs:
    model: str = ""
    dataset: str = "dataset/countdown/test.parquet"
    num_generation: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 1024
    start_pos: int = 0
    end_pos: int = 1024
    batch_size: int = 16
    out_dir: str = "outputs/diverse"
    calc_path: list[str] = field(default_factory=list)
    method: str = "our"  # old
    run_name: str = ""

    rollout_filter_edit_dist_thres: float | None = 0.4
    rollout_filter_suffix_match_thres: float | None = None
    rollout_filter_rouge_l_thres: float | None = None
    prob_filter_abs_thres: float = 0.25
    prob_filter_rel_thres: float = 0.15
    rollout_filter_steps: list[int] = field(default_factory=lambda: [20, 30, 50])


class EvalCollator:
    def __init__(self, tokenizer, template):
        self.tokenizer = tokenizer
        self.template = template

    def __call__(self, batch: list[dict[str, Any]]):
        prompts = [make_prompt_countdown(sample, self.template) for sample in batch]
        encoding = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        return {
            "input_ids": encoding["input_ids"].cuda(),
            "attention_mask": encoding["attention_mask"].cuda(),
            "target": [sample["target"] for sample in batch],
            "nums": [sample["nums"] for sample in batch],
        }


def group_batch_list(x: list, group_size: int):
    assert len(x) % group_size == 0
    return [x[i : i + group_size] for i in range(0, len(x), group_size)]


def inference(args: EvalArgs):
    assert args.model, "model is required"
    print(f"running inference from {args.start_pos} to {args.end_pos}")
    model_name = os.path.basename(args.model)
    filename = f"{args.start_pos}_{args.end_pos}.jsonl"
    dir = os.path.join(args.out_dir, model_name + "_" + args.method)
    if args.run_name:
        dir += f"_{args.run_name}"

    if args.start_pos == 0:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).cuda()
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    dataset = Dataset.from_parquet(args.dataset)
    if args.end_pos > 0:
        dataset = dataset.select(range(args.start_pos, args.end_pos))  # type: ignore

    template = "inst" if "inst" in model_name.lower() else "base"
    collator = EvalCollator(tokenizer, template)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)  # type: ignore
    gen_config = KeyTokenGenConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_return_sequences=args.num_generation,
        return_on_full=False,
        sample_nk="full",
        rollout_filter_edit_dist_thres=args.rollout_filter_edit_dist_thres,
        rollout_filter_suffix_match_thres=args.rollout_filter_suffix_match_thres,
        rollout_filter_rouge_l_thres=args.rollout_filter_rouge_l_thres,
        prob_filter_abs_thres=args.prob_filter_abs_thres,
        prob_filter_rel_thres=args.prob_filter_rel_thres,
        rollout_filter_steps=args.rollout_filter_steps,
    )
    kt_modules = KtModules()
    group_metrics = []

    for batch in tqdm(loader):
        input_len = batch["input_ids"].shape[1]
        if args.method == "our":
            outputs = generate(
                model, kt_modules, batch["input_ids"], tokenizer, batch["attention_mask"], config=gen_config
            )
        else:
            outputs = model.generate(
                batch["input_ids"].repeat_interleave(args.num_generation, dim=0),
                attention_mask=batch["attention_mask"].repeat_interleave(args.num_generation, dim=0),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k if args.top_k > 0 else None,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                return_dict_in_generate=True,
            )
        sequences = tokenizer.batch_decode(outputs.sequences[:, input_len:], skip_special_tokens=True)
        answer_strs, answer_values = [], []
        for seq in sequences:
            match = re.search(r"<answer>(.*?)</answer>", seq)
            if match:
                answer_str = match.group(1).split("=", 1)[0].strip()
                answer_strs.append(answer_str)
                answer_values.append(eval_countdown_equation(answer_str))
            else:
                answer_strs.append("")
                answer_values.append(-100)
        sequences = group_batch_list(sequences, args.num_generation)
        answer_strs = group_batch_list(answer_strs, args.num_generation)
        answer_values = group_batch_list(answer_values, args.num_generation)
        for seqs, strs, vals, target, nums in zip(
            sequences, answer_strs, answer_values, batch["target"], batch["nums"], strict=True
        ):
            num_str = len(set(map(lambda x: x.replace(" ", ""), strs)))
            num_val = len(set(vals))
            bleu, rouge = pairwise_similarity(seqs)
            pass_1 = sum(
                1 for v, s in zip(vals, strs) if v == target and validate_countdown_equation(s, nums)
            ) / len(vals)
            pass_k = int(pass_1 > 0)
            add_info = (
                {
                    "saturate_len": outputs.avg_saturate_len,
                    "branching_ratio": outputs.branching_ratio,
                    "pruning_ratio": outputs.pruning_ratio,
                }
                if args.method == "our"
                else {}
            )
            group_metrics.append(
                {
                    "generations": [
                        {"completion": seq, "answer_extracted": s, "answer_value": v}
                        for seq, s, v in zip(seqs, strs, vals, strict=True)
                    ],
                    "num_str": num_str,
                    "num_val": num_val,
                    "bleu": bleu,
                    "rouge": rouge,
                    "pass_1": pass_1,
                    "pass_k": pass_k,
                    **add_info,
                }
            )

    print(f"saving {filename} to {dir}")
    with open(os.path.join(dir, filename), "w") as f:
        for metric in group_metrics:
            f.write(json.dumps(metric) + "\n")


def get_sort_key(x: tuple[str, dict[str, Any]]):
    step, run = x[0].split("_", 1)
    return run, step


def calculate(args: EvalArgs):
    name_values = defaultdict(dict)
    keys = None
    for path in args.calc_path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")
        files = glob(os.path.join(path, "*.jsonl"))
        if not files:
            continue
        name = os.path.basename(path)
        metrics = []
        for file in files:
            with open(file, "r") as f:
                metrics.extend([json.loads(line) for line in f])
        keys = list(metrics[0].keys())
        keys.remove("generations")
        for key in keys:
            values = sum(metric[key] for metric in metrics) / len(metrics)
            print(f"{name} {key}: {values:.4f}")
            name_values[name][key] = values

    assert keys is not None

    with open(os.path.join(args.out_dir, "metrics.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["name"] + keys)
        for name, values in natsorted(name_values.items(), key=get_sort_key):
            writer.writerow([name] + [values.get(key, "") for key in keys])


if __name__ == "__main__":
    parser = HfArgumentParser([EvalArgs])  # type: ignore
    args = cast(EvalArgs, parser.parse_args_into_dataclasses()[0])
    if args.calc_path:
        calculate(args)
    else:
        inference(args)
