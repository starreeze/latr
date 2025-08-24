"compare the diversity of the key token model and the explicit generation model"

import json
import os
import warnings
from datetime import datetime
from itertools import combinations
from typing import Any, cast

import nltk
import numpy as np
import torch
from datasets import Dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser

from args import DataArgs, GenerationArgs, ModelArgs, TrainingArgs
from data import eval as answer_processor
from data.loader import get_sft_collator
from data.loader import load as load_datasets
from model.generation import KeyTokenGenMixin
from model.loader import load as load_model
from tools.multiprocess import MultiGPUManager, MultiGPUWorker
from tools.utils import set_seed, to_device

# Download required NLTK data (run once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def pairwise_similarity(
    inputs: list[str],
    bleu_weights: tuple[float, float, float, float] = (0, 0, 0.5, 0.5),
    smoothing_method: str = "method1",
    rouge_types: list[str] = ["rougeL"],
    use_stemmer: bool = False,
) -> tuple[float, float] | tuple[None, None]:
    """
    Calculate average pairwise BLEU and ROUGE metrics for a group of sentences.

    Args:
        inputs : List[str]
            List of sentences to compare pairwise
        bleu_weights : Tuple[float, float, float, float], default=(0.25, 0.25, 0.25, 0.25)
            Weights for n-gram precisions (1-gram to 4-gram)
            Examples:
            - (1, 0, 0, 0): Only 1-gram precision
            - (0.5, 0.5, 0, 0): Equal weight for 1-gram and 2-gram
            - (0.25, 0.25, 0.25, 0.25): Geometric mean (default)
        smoothing_method : str, default='method1'
            Smoothing method for BLEU scores to handle zero counts
            Options: 'method0', 'method1', 'method2', ..., 'method7'
        rouge_types : List[str], default=['rouge1', 'rouge2', 'rougeL']
            Types of ROUGE scores to calculate
            Options: 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'
        use_stemmer : bool, default=False
            Whether to use Porter stemmer in ROUGE calculation

    Returns:
    Tuple[float, float]
        (average_bleu_score, average_rouge_f1_score)
    """

    if len(inputs) < 2:
        return None, None

    # Initialize metrics
    total_bleu = 0.0
    total_rouge_f1 = 0.0
    pair_count = 0

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=use_stemmer)

    # Calculate pairwise metrics
    for sentence1, sentence2 in combinations(inputs, 2):
        # Calculate BLEU score
        reference = [sentence1.split()]
        candidate = sentence2.split()

        try:
            bleu_score = sentence_bleu(
                reference,
                candidate,
                weights=bleu_weights,
                smoothing_function=getattr(SmoothingFunction(), smoothing_method),
            )
            assert isinstance(bleu_score, float | int)
        except Exception as e:
            warnings.warn(f"BLEU calculation failed for pair: {e}")
            bleu_score = 0.0

        # Calculate ROUGE scores
        try:
            rouge_scores = scorer.score(sentence1, sentence2)
            # Average F1 scores across all rouge types
            rouge_f1_avg = sum(score.fmeasure for score in rouge_scores.values()) / len(rouge_scores)
        except Exception as e:
            warnings.warn(f"ROUGE calculation failed for pair: {e}")
            rouge_f1_avg = 0.0

        total_bleu += bleu_score
        total_rouge_f1 += rouge_f1_avg
        pair_count += 1

    # Calculate averages
    avg_bleu = total_bleu / pair_count if pair_count > 0 else 0.0
    avg_rouge_f1 = total_rouge_f1 / pair_count if pair_count > 0 else 0.0

    return avg_bleu, avg_rouge_f1


class EvalWorker(MultiGPUWorker):
    """Worker class for evaluation across multiple GPUs."""

    def process_chunk(self, dataset_chunk: Dataset, gpu_id: int, device: str, **kwargs) -> list:
        """Process a chunk of dataset for evaluation."""
        model_args: ModelArgs = kwargs["model_args"]
        data_args: DataArgs = kwargs["data_args"]
        generation_args: GenerationArgs = kwargs["generation_args"]

        print(f"Worker {gpu_id}: Loading model to {device}")
        model, tokenizer = load_model(model_args)
        model.to(device)  # type: ignore
        model.eval()

        # Set up data processing
        data_args.align_label_with_tf_mask = False
        collator = get_sft_collator(tokenizer, data_args, keep_original_data=True)
        test_loader = DataLoader(dataset_chunk, batch_size=data_args.generate_batch_size, collate_fn=collator)  # type: ignore
        answer_extractor = answer_processor.AnswerExtractor(data_args)
        answer_matcher = getattr(answer_processor, f"{data_args.eval_dataset.capitalize()}AnswerMatcher")()
        answer_simplifier = getattr(
            answer_processor, f"{data_args.eval_dataset.capitalize()}AnswerSimplifier"
        )()

        print(f"Worker {gpu_id}: Evaluating {len(dataset_chunk)} samples")

        results_list = []

        for batch in tqdm(
            test_loader,
            total=len(test_loader),
            desc=f"eval {data_args.eval_dataset} GPU {gpu_id}",
            position=gpu_id,
        ):
            batch: dict[str, Any] = to_device(batch, device)  # type: ignore
            question_ids = batch["question_ids"]
            attention_mask = batch["attention_mask"]
            question_strs = tokenizer.batch_decode(question_ids, skip_special_tokens=True)

            kwargs_gen = generation_args.__dict__
            if isinstance(model, KeyTokenGenMixin):
                res = model.generate(
                    question_ids, tokenizer=tokenizer, attention_mask=attention_mask, **kwargs_gen
                )
                pred_raw: torch.Tensor = res.sequences
                assert res.num_seq is not None
                num_seq = res.num_seq / len(question_ids)
            else:
                new_keys = ["temperature", "top_k", "top_p", "do_sample", "use_cache", "max_new_tokens"]
                new_kwargs = {k: kwargs_gen[k] for k in new_keys if k in kwargs_gen}
                pred_raw = model.generate(
                    question_ids,
                    attention_mask=attention_mask,
                    **new_kwargs,
                    num_return_sequences=generation_args.num_return_sequences,
                )  # type: ignore
                num_seq = 0

            # instead of truncating it here, we handle it after decoding since explicit generation
            # strategy may add extra tokens and disturb the sequence
            # pred_raw = pred_raw[:, question_raw.shape[1] :]
            pred_raw_strs = tokenizer.batch_decode(pred_raw, skip_special_tokens=True)
            interval = generation_args.num_return_sequences
            pred_raw_strs = [
                pred_raw_strs[i * interval : (i + 1) * interval] for i in range(len(question_ids))
            ]

            batch_flattened = [{k: v[i] for k, v in batch.items()} for i in range(len(question_ids))]
            for sample, question_str, preds_raw in zip(batch_flattened, question_strs, pred_raw_strs):
                responses = []
                for pred_raw_str in preds_raw:
                    assert pred_raw_str.startswith(question_str)
                    pred_raw_str = pred_raw_str[len(question_str) :]
                    pred_extracted = answer_extractor(pred_raw_str)
                    is_correct = answer_matcher(pred_extracted, sample)
                    simplified = answer_simplifier(pred_extracted)
                    result = {
                        "pred_raw": pred_raw_str,
                        "pred_extracted": pred_extracted,
                        "is_correct": is_correct,
                        "simplified": simplified,
                    }
                    responses.append(result)

                bleu_score, rouge_f1 = pairwise_similarity([r["pred_raw"] for r in responses])
                results = {
                    "question": question_str,
                    "truth": sample["answer"],
                    "num_seq": num_seq,
                    "num_diff_answers": len(set([r["simplified"] for r in responses])),
                    "bleu_score": bleu_score,
                    "rouge_f1": rouge_f1,
                    "responses": responses,
                }
                results_list.append(results)

        print(f"Worker {gpu_id}: Finished evaluation")
        return results_list

    def combine_results(self, results: list[list]) -> list[dict[str, Any]]:
        """Combine results from multiple workers."""
        return sum(results, [])


def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainingArgs, GenerationArgs))  # type: ignore
    model_args, data_args, train_args, generation_args = parser.parse_args_into_dataclasses()
    model_args = cast(ModelArgs, model_args)
    data_args = cast(DataArgs, data_args)
    train_args = cast(TrainingArgs, train_args)
    generation_args = cast(GenerationArgs, generation_args)

    set_seed(train_args.seed)

    if generation_args.max_think_tokens is not None:
        raise NotImplementedError("max_think_tokens is not supported for key token generation")

    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    model_name = os.path.basename(model_args.model.rstrip("/"))
    type = "key_token" if model_args.force_key_token_model is not None else "explicit"
    run_name = f"{model_name}_{data_args.eval_dataset}_{type}_{timestamp}"
    eval_dir = os.path.join(os.path.dirname(train_args.output_dir), "eval")
    dir_path = os.path.join(eval_dir, run_name)
    results_file = os.path.join(dir_path, "results.jsonl")
    config_file = os.path.join(dir_path, "config.json")

    _, test_dataset = load_datasets(data_type="sft", **data_args.__dict__)
    if data_args.end_pos < len(test_dataset) or data_args.start_pos > 0:
        print(f"Selecting {data_args.start_pos} to {data_args.end_pos} from {len(test_dataset)} samples")
        test_dataset = test_dataset.select(range(data_args.start_pos, data_args.end_pos))

    n_gpus = data_args.n_gpus
    print(f"Using {n_gpus} GPUs for evaluation.")

    # Use the multiprocessing framework for all cases (single and multi-GPU)
    manager = MultiGPUManager(EvalWorker)
    eval_results: list[dict[str, Any]] = manager.run(
        test_dataset, n_gpus, model_args=model_args, data_args=data_args, generation_args=generation_args
    )

    os.makedirs(dir_path, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(vars(model_args) | vars(generation_args) | vars(data_args), f, indent=2)

    # Write results to file
    with open(results_file, "w") as f:
        for result in eval_results:
            f.write(f"{json.dumps(result)}\n")

    # Calculate metrics
    total_responses = 0
    correct_responses = 0
    questions_with_correct_response = 0
    total_questions = len(eval_results)
    total_num_diff_answers = 0
    total_bleu = 0.0
    total_rouge = 0.0
    total_std = 0.0
    total_num_seq = 0
    non_single_count = 0

    for result in eval_results:
        results = []
        # Count total responses and correct responses for overall accuracy
        for response in result["responses"]:
            results.append(int(response["is_correct"]))
        total_responses += len(results)
        correct_responses += sum(results)
        total_num_seq += result["num_seq"]
        total_num_diff_answers += result["num_diff_answers"]

        if len(results) > 1:
            non_single_count += 1
            total_std += np.std(results)
            assert result["bleu_score"] is not None
            assert result["rouge_f1"] is not None
            total_bleu += result["bleu_score"]
            total_rouge += result["rouge_f1"]
        else:
            assert result["bleu_score"] is None
            assert result["rouge_f1"] is None

        # Check if at least one response is correct for pass@1
        if any(results):
            questions_with_correct_response += 1

    # Calculate final metrics
    overall_accuracy = (correct_responses / total_responses * 100) if total_responses > 0 else 0.0
    pass_at_1 = (questions_with_correct_response / total_questions * 100) if total_questions > 0 else 0.0
    avg_num_seq = total_num_seq / total_questions if total_questions > 0 else 0.0
    avg_num_diff_answers = total_num_diff_answers / total_questions if total_questions > 0 else 0.0

    avg_bleu = total_bleu / non_single_count if non_single_count > 0 else 0.0
    avg_rouge = total_rouge / non_single_count if non_single_count > 0 else 0.0
    avg_std = total_std / non_single_count if non_single_count > 0 else 0.0

    # Print results
    print(f"Total Questions: {total_questions}")
    print(f"Total Responses: {total_responses}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}% ({correct_responses}/{total_responses})")
    print(f"Pass@k: {pass_at_1:.1f}% ({questions_with_correct_response}/{total_questions})")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE-F1: {avg_rouge:.4f}")
    print(f"Average Std: {avg_std:.4f}")
    print(f"Average Number of Sequences: {avg_num_seq:.4f}")
    print(f"Average Number of Different Answers: {avg_num_diff_answers:.4f}")

    # Update score file format and content
    score_file = os.path.join(eval_dir, data_args.score_file)
    if not os.path.exists(score_file):
        with open(score_file, "w") as f:
            f.write(
                "name,total_questions,total_responses,overall_accuracy,"
                "pass_at_k,avg_bleu,avg_rouge,avg_std,avg_num_seq,avg_num_diff_answers\n"
            )

    with open(score_file, "a") as f:
        f.write(
            f"{run_name},{total_questions},{total_responses},{overall_accuracy:.1f},"
            f"{pass_at_1:.1f},{avg_bleu:.4f},{avg_rouge:.4f},{avg_std:.4f},"
            f"{avg_num_seq:.2f},{avg_num_diff_answers:.2f}\n"
        )


if __name__ == "__main__":
    main()
