# -*- coding: utf-8 -*-
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, cast

import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils import PreTrainedTokenizer

from data.base import DataBase
from data.prompts import PromptFormatter
from tools.utils import find_values


def filter_coco_stage_data(
    data: Dataset,
    target_column: str,
    step_fn: Callable[[Any], int],
    current_step: int,
    data_mixing_ratio: float,
) -> Dataset:
    """
    Filter and mix dataset samples based on complexity stages for curriculum learning.

    This function implements a curriculum learning strategy where training data is filtered
    based on the number of steps/complexity. It includes all samples at or above the current
    stage complexity, plus a proportion of simpler samples for mixing.

    Args:
        data: The input dataset to filter
        target_column: Name of the column containing the data to measure complexity from
        step_fn: Function that takes an item from target_column and returns the number of steps/complexity
        current_stage: The current complexity stage (minimum number of steps to include all samples).
                      If < 0, no filtering is applied.
        data_mixing_ratio: Ratio of simpler samples (steps < current_stage) to include relative to
                          complex samples (steps >= current_stage). If < 0, no filtering is applied.

    Returns:
        Dataset: Filtered and shuffled dataset containing:
                - All samples with steps >= current_stage
                - A sampled subset of simpler samples (steps < current_stage) based on data_mixing_ratio

    Examples:
        If current_stage=3 and data_mixing_ratio=1:
        - Include all samples with 3+ steps
        - Include as many samples with <3 steps as 3+ steps (randomly sampled)

    Note:
        - If current_stage < 0 or data_mixing_ratio < 0, returns original data unchanged
        - The returned dataset is shuffled to mix complex and simple samples
        - If there are fewer simpler samples than needed, all available simpler samples are used
    """
    if current_step < 0 or data_mixing_ratio < 0:
        return data

    num_steps = list(map(step_fn, data[target_column]))
    num_step_to_idx = [[] for _ in range(max(num_steps) + 1)]
    for i, num_step in enumerate(num_steps):
        num_step_to_idx[num_step].append(i)
    longer_idx = sum(num_step_to_idx[current_step:], [])
    shorter_idx = sum(num_step_to_idx[:current_step], [])
    sample_num = min(int(len(longer_idx) * data_mixing_ratio), len(shorter_idx))
    sampled_shorter_idx = random.sample(shorter_idx, sample_num)
    return data.select(longer_idx + sampled_shorter_idx).shuffle()


def get_gsm8k_cot_answer(text: str, cot_content: str) -> tuple[list[str], str]:
    raw_cot, answer_str = text.split("####")
    raw_cot = raw_cot.strip()
    answer_str = answer_str.strip().replace(",", "")

    step_regex = re.compile(r"\$?(<<.+?>>)")  # Non-greedy match for content within << >>
    matches = list(step_regex.finditer(raw_cot))

    if cot_content == "formula":
        # a step only contains one formula
        steps = [match.group(1) for match in matches]
        return steps, answer_str
    assert cot_content == "sentence"

    # a step contain the complete sentence
    steps = []
    current_search_start_in_cot = 0

    for i, match in enumerate(matches):
        # Determine the search boundaries for the step's end delimiter
        delimiter_search_start = match.end()
        if i < len(matches) - 1:
            delimiter_search_end = matches[i + 1].start()
        else:
            delimiter_search_end = len(raw_cot)

        # Find the first '.' or '\\n' in the suffix part of the step
        suffix_text = raw_cot[delimiter_search_start:delimiter_search_end]
        period_idx = suffix_text.find(".")
        newline_idx = suffix_text.find("\n")

        actual_step_end_in_cot = 0
        if period_idx != -1 and (newline_idx == -1 or period_idx < newline_idx):
            # Period is the delimiter
            actual_step_end_in_cot = delimiter_search_start + period_idx + 1
        elif newline_idx != -1:
            # Newline is the delimiter
            actual_step_end_in_cot = delimiter_search_start + newline_idx + 1
        else:
            # No delimiter found, step ends with the suffix_text
            actual_step_end_in_cot = delimiter_search_start + len(suffix_text)

        step = raw_cot[current_search_start_in_cot:actual_step_end_in_cot]

        # Clean the step
        step = re.sub(step_regex, "", step.strip().replace("\n", ""))
        step = (step if step.endswith(".") else step + ".") + (" " if i < len(matches) - 1 else "")
        steps.append(step)

        # Update start for the next step search to be after the current step's delimiter
        current_search_start_in_cot = actual_step_end_in_cot
        # Skip any immediate whitespace for the next step's start
        while current_search_start_in_cot < len(raw_cot) and raw_cot[current_search_start_in_cot].isspace():
            current_search_start_in_cot += 1

    return steps, answer_str


class SFTData(DataBase):
    def process_countdown_sample(self, x: dict) -> dict:
        plain_prompt = self.prompt_formatter(numbers=x["nums"], target=x["target"])
        message_prompt = [*self.system_prompt_message, {"role": "user", "content": plain_prompt}]
        return {
            "question": message_prompt if self.template == "conv" else plain_prompt,
            "cot": [x.get("cot", "")],
            "answer": x.get("answer", ""),
            "difficulty": x.get("difficulty", 0.0),
            "numbers": x["nums"],
            "target": x["target"],
        }

    def process_gsm8k_sample(self, x: dict) -> dict:
        raise NotImplementedError("gsm8k is not supported")

    def process_aime_sample(self, x: dict) -> dict:
        plain_prompt = self.prompt_formatter(question=x["question"])
        message_prompt = [*self.system_prompt_message, {"role": "user", "content": plain_prompt}]
        return {
            "question": message_prompt if self.template == "conv" else plain_prompt,
            "answer": x["answer"],
            "cot": [x.get("cot", "")],
            "difficulty": x.get("difficulty", 0.0),
        }

    def process_dapomath_sample(self, x: dict) -> dict:
        plain_prompt = self.prompt_formatter(question=x["prompt"])
        message_prompt = [*self.system_prompt_message, {"role": "user", "content": plain_prompt}]
        return {
            "question": message_prompt if self.template == "conv" else plain_prompt,
            "answer": x["solution"],
            "cot": [x.get("cot", "")],
            "difficulty": x.get("difficulty", 0.0),
        }

    def process_math500_sample(self, x: dict) -> dict:
        plain_prompt = self.prompt_formatter(question=x["problem"])
        message_prompt = [*self.system_prompt_message, {"role": "user", "content": plain_prompt}]
        return {
            "question": message_prompt if self.template == "conv" else plain_prompt,
            "answer": x["answer"],
            "cot": [x.get("solution", "")],
            "difficulty": x.get("difficulty", 0.0),
        }


class SFTCollator(ABC):
    all_answer_prefix = {"qwen2": "<|im_start|>assistant\n", "llama": "<｜Assistant｜>"}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        align_label_with_tf_mask=True,
        keep_original_data=False,
        cot_type="none",
        template="conv",
        system_prompt_type="general",
        train=False,
        latent_model=False,
        n_repeat: int | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.eos_token = cast(str, tokenizer.eos_token)
        self.eos_token_id = cast(int, tokenizer.eos_token_id)
        self.pad_token_id = cast(int, tokenizer.pad_token_id)
        self.align_label_with_tf_mask = align_label_with_tf_mask
        self.keep_original_data = keep_original_data
        self.cot_type = cot_type
        self.template = template
        self.system_prompt_message = PromptFormatter.get_system_prompt(system_prompt_type)
        self.train = train
        self.latent_model = latent_model
        self.n_repeat = n_repeat

        tok_bb_name = tokenizer.__class__.__name__.lower()
        for t in self.all_answer_prefix.keys():
            if t in tok_bb_name:
                self.answer_prefix = self.all_answer_prefix[t]
                break
        else:
            raise ValueError(f"Cannot find answer prefix for tokenizer class {tok_bb_name}")

    def _get_texts(self, batch: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        questions = [sample["question"] for sample in batch]
        answers = [
            self._construct_ans_str(sample["cot"], sample["answer"], sample["difficulty"]) for sample in batch
        ]

        if self.template == "plain":
            question_texts = questions
            all_texts = [
                question + " " + answer + self.eos_token for question, answer in zip(questions, answers)
            ]
            return question_texts, all_texts

        assert self.template == "conv"
        question_texts = []
        all_texts = []

        for question, answer in zip(questions, answers):
            # Build messages for this sample
            prefix_message = question
            all_message = prefix_message + [{"role": "assistant", "content": answer}]

            # get question text
            question_text = cast(str, self.tokenizer.apply_chat_template(prefix_message, tokenize=False))
            question_text += self.answer_prefix
            question_texts.append(question_text)

            # get all text
            all_text = cast(str, self.tokenizer.apply_chat_template(all_message, tokenize=False))
            all_text = all_text.strip()
            if not all_text.endswith(self.eos_token):
                all_text = all_text + self.eos_token
            all_texts.append(all_text)

            assert len(question_text) < len(all_text) and all_text[: len(question_text)] == question_text
        return question_texts, all_texts

    def _pad_returns(self, q_tokens, all_tokens, all_labels, all_tf_masks):
        padding_side = "right" if self.train and self.latent_model else "left"
        questions = pad_sequence(
            q_tokens, batch_first=True, padding_value=self.pad_token_id, padding_side=padding_side
        )
        if not self.train:
            return {"question_ids": questions, "attention_mask": questions != self.pad_token_id}

        tokens = pad_sequence(
            all_tokens, batch_first=True, padding_value=self.pad_token_id, padding_side=padding_side
        )
        labels = pad_sequence(all_labels, batch_first=True, padding_value=-100, padding_side=padding_side)
        if all_tf_masks:
            teacher_forcing_mask = pad_sequence(all_tf_masks, batch_first=True, padding_value=1)
        else:
            teacher_forcing_mask = None
        attention_mask = tokens != self.pad_token_id

        # for cases where pad is the same as eos, we need to set the attention mask to True
        # at the actual eos position
        if self.pad_token_id == self.eos_token_id:
            effective_len = find_values(attention_mask, True, reverse=True) + 1
            assert effective_len.max().item() < attention_mask.shape[1], "eos not found"
            attention_mask[torch.arange(effective_len.shape[0]), effective_len] = True

        return {
            "question_ids": questions,
            "input_ids": tokens,
            "labels": labels,
            "teacher_forcing_mask": teacher_forcing_mask,
            "attention_mask": attention_mask,
        }

    @abstractmethod
    def _construct_ans_str(self, cot: list[str], answer: str, difficulty: float) -> str:
        pass

    def _get_tokens_labels(
        self, q_text: str, all_text: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_token = self.tokenizer([q_text], return_tensors="pt", add_special_tokens=False).input_ids[0]
        q_len = len(q_token)
        all_token = self.tokenizer([all_text], return_tensors="pt", add_special_tokens=False).input_ids[0]
        label = all_token.clone()
        label[:q_len] = -100
        return q_token, all_token, label

    @abstractmethod
    def _update_labels_tf_mask(
        self, tokens: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pass

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if self.n_repeat is not None:
            assert len(batch) == 1, "n_repeat is only supported for batch size 1"
            batch = batch * self.n_repeat

        question_texts, all_texts = self._get_texts(batch)

        all_labels = []
        q_tokens = []
        all_tokens = []
        all_tf_masks = []
        for all_text, q_text in zip(all_texts, question_texts):
            q_token, all_token, labels = self._get_tokens_labels(q_text, all_text)
            labels, tf_mask = self._update_labels_tf_mask(all_token, labels)
            q_tokens.append(q_token)
            all_tokens.append(all_token)
            all_labels.append(labels)
            if tf_mask is not None:
                all_tf_masks.append(tf_mask)

        res = self._pad_returns(q_tokens, all_tokens, all_labels, all_tf_masks)

        if self.keep_original_data:
            res |= {k: [sample[k] for sample in batch] for k in batch[0].keys()}
        return res


class ExplicitSFTCollator(SFTCollator):
    def __init__(self, tokenizer: PreTrainedTokenizer, align_label_with_tf_mask=True, **kwargs) -> None:
        super().__init__(tokenizer, align_label_with_tf_mask, **kwargs)

    def _construct_ans_str(self, cot: list[str], answer: str, difficulty: float) -> str:
        cot_str = " ".join(cot)
        if self.cot_type == "none":
            return answer
        elif self.cot_type == "plain":
            return f"<think>\n{cot_str}\n</think>\n\\boxed{{{answer}}}"
        elif self.cot_type == "xml":
            return f"<think>\n{cot_str}\n</think>\n<answer>\n{answer}\n</answer>"
        else:
            raise ValueError(f"Invalid cot_type: {self.cot_type} for explicit sft")

    def _update_labels_tf_mask(
        self, tokens: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return labels, None
