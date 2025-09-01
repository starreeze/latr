from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Iterable, Optional, cast

import nltk
import torch
from torchaudio.functional import edit_distance
from transformers.tokenization_utils import PreTrainedTokenizer

from model.diverge import DivCollator, DivJudge
from model.utils import BranchInfo

math_core_symbols = "+-*/\\<>=()[]{}_^&%$0123456789"


class KeyTokenFilter(ABC):
    "filters that determine whether new branches should be sprinted up at a specific token"

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(
        self, sequences: torch.Tensor, candidates: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Judge whether this is a key token. If so, return the candidates. Otherwise it is replaced with -1.
        sequence: (seq_len,)
        candidates: (max_n_branch,)
        probs: (max_n_branch,)
        Return: mask (max_n_branch,)
        """
        pass


class ProbFilter(KeyTokenFilter):
    "only keep tokens with probability greater than prob_filter_thres"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prob_filter_abs_thres: float | None,
        prob_filter_rel_thres: float | None,
    ):
        super().__init__(tokenizer)
        self.prob_filter_abs_thres = prob_filter_abs_thres
        self.prob_filter_rel_thres = prob_filter_rel_thres

    def __call__(
        self, sequences: torch.Tensor, candidates: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        bs, n_cand = probs.shape
        mask = torch.ones([bs, n_cand - 1], dtype=torch.bool, device=probs.device)
        if self.prob_filter_abs_thres is not None:
            mask = mask & (probs[:, 1:] > self.prob_filter_abs_thres)
        if self.prob_filter_rel_thres is not None:
            mask = mask & (probs[:, 0:1] - probs[:, 1:] > self.prob_filter_rel_thres)
        return mask


class SimilarityFilter(KeyTokenFilter):
    "only keep remaining tokens with similarity to the first token less than similarity_filter_thres"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        embedding_module: torch.nn.Module,
        similarity_filter_thres: float,
    ):
        raise NotImplementedError("SimilarityFilter is not implemented")
        # NOTE align shape if enabled
        super().__init__(tokenizer)
        self.similarity_filter_thres = similarity_filter_thres
        self.embedding_module = embedding_module

    def __call__(
        self, sequences: torch.Tensor, candidates: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("SimilarityFilter is not implemented")
        # Filter out negative values (padding) to get valid candidates
        valid_mask = candidates >= 0
        valid_candidates = candidates[valid_mask]

        # If there's only one valid candidate or no valid candidates, return as is
        if len(valid_candidates) <= 1:
            return candidates

        # Get embeddings only for valid candidates
        embeddings = self.embedding_module(valid_candidates)
        first_embeds = embeddings[0].unsqueeze(0)
        remaining_embeds = embeddings[1:]
        similarities = torch.cosine_similarity(first_embeds, remaining_embeds, dim=-1)

        # Create mask for all candidates
        similarity_mask = similarities < self.similarity_filter_thres
        first_mask = torch.ones_like(candidates[:, 0:1], dtype=torch.bool)
        return torch.cat([first_mask, similarity_mask], dim=1)


class StopWordFilter(KeyTokenFilter):
    "only keep tokens that are not stop words, punctuations, spaces, etc."

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer)
        # Get English stop words from NLTK
        try:
            self.stop_words = set(nltk.corpus.stopwords.words("english"))
        except LookupError:
            # Download stopwords if not available
            nltk.download("stopwords")
            self.stop_words = set(nltk.corpus.stopwords.words("english"))

        # Punctuation to filter out
        self.filter_punctuation = ".,;:!?\"'"

    def __call__(
        self, sequences: torch.Tensor, candidates: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        batch, n_cand = candidates.shape
        mask = torch.ones([batch, n_cand - 1], dtype=torch.bool)

        for sample_idx, candidate in enumerate(candidates):
            for token_idx, token_id in enumerate(candidate[1:]):
                token_id = int(token_id.item())
                # keep the first token, otherwise stop words will never be sampled out
                if token_id == -1:
                    continue

                token_text = self.tokenizer.decode([token_id]).strip()
                # Handle subword tokens (remove common prefixes)
                clean_text = token_text
                if clean_text.startswith("▁"):  # SentencePiece
                    clean_text = clean_text[1:]
                elif clean_text.startswith("##"):  # BERT-style
                    clean_text = clean_text[2:]

                if clean_text.lower() in self.stop_words:
                    mask[sample_idx, token_idx] = False
                    continue
                if clean_text and all(c in self.filter_punctuation for c in clean_text):
                    mask[sample_idx, token_idx] = False
                    continue
                if not clean_text or clean_text.isspace():
                    mask[sample_idx, token_idx] = False
                    continue

        return mask.to(candidates.device)


class ModelFilter(KeyTokenFilter):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        filter_model: DivJudge,
        model_filter_thres: float,
        keep_math_core_symbols: bool = False,
    ):
        super().__init__(tokenizer)
        self.filter_model = filter_model
        self.model_filter_thres = model_filter_thres
        self.collator = DivCollator(tokenizer)
        self.keep_math_core_symbols = keep_math_core_symbols

    def __call__(
        self, sequences: torch.Tensor, candidates: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        bs, n_cand = candidates.shape
        batch = [
            {"prefix": sequence, "a": candidate[0], "b": candidate[i]}
            for sequence, candidate in zip(sequences, candidates)
            for i in range(1, n_cand)
        ]
        with torch.no_grad():
            scores = self.filter_model(**self.collator(batch)).scores

        if self.keep_math_core_symbols:
            ids = candidates.reshape(-1).cpu().tolist()
            for id in ids:
                token = self.tokenizer.decode([id]).strip()
                if token in math_core_symbols:
                    scores[id] = 0

        scores = scores.reshape(bs, n_cand - 1)
        return scores < self.model_filter_thres


class KeyTokenFilterList:
    def __init__(self, filters: Iterable[KeyTokenFilter]):
        self.filters = list(filters)

    def __call__(
        self, sequences: torch.Tensor, candidates: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        "return shape: (batch, n_cand) with the first token always kept"
        batch, n_cand = candidates.shape
        mask = torch.ones([batch, n_cand - 1], device=candidates.device, dtype=torch.bool)
        for filter in self.filters:
            mask = mask & filter(sequences, candidates, probs)
            if not mask.any():
                break
        mask = torch.cat([torch.ones([batch, 1], device=candidates.device, dtype=torch.bool), mask], dim=1)
        return mask

    def append(self, filter: KeyTokenFilter):
        self.filters.append(filter)

    def __len__(self) -> int:
        return len(self.filters)

    def __getitem__(self, idx):
        return self.filters[idx]


class SequenceFilter(ABC):
    "sequence based filter: determine whether a sequence should be deleted"

    @abstractmethod
    def __call__(self, sequences: torch.Tensor, branch_info: BranchInfo, current_step: int) -> set[int]:
        "return remove indices"
        pass


class RolloutFilter(SequenceFilter):
    """
    Rollout filter that applies edit distance comparison between sequences from sibling branches.
    """

    def __init__(
        self,
        rollout_filter_steps: list[int],
        rollout_filter_edit_dist_thres: float | None,
        model_filter_thres: float | None,
        tokenizer: PreTrainedTokenizer,
        filter_model: DivJudge | None,
        keep_math_core_symbols: bool = False,
    ):
        self.rollout_filter_steps = rollout_filter_steps
        self.rollout_filter_edit_dist_thres = rollout_filter_edit_dist_thres
        self.model_filter_thres = model_filter_thres
        self.filter_model = filter_model
        self.tokenizer = tokenizer
        self.collator = DivCollator(tokenizer)
        self.keep_math_core_symbols = keep_math_core_symbols
        self.eos_id = cast(int, tokenizer.eos_token_id)

    def __call__(self, completion_ids: torch.Tensor, branch_info: BranchInfo, current_step: int) -> set[int]:
        """
        Apply rollout filter by comparing edit distances and model scores of sequences from branches
        created `rollout_filter_steps` steps ago.

        Returns a single set of sequence indices (w.r.t. current batch ordering) to remove.
        """
        remove_set: set[int] = set()
        model_batch = []
        meta: list[tuple[int, int]] = []  # (branch_idx, length)

        target_branch_ids = []
        for length in self.rollout_filter_steps:
            target_step = current_step - length
            target_branch_ids.extend(branch_info.get_branch_ids_by_birth_step(target_step))

        for idx in target_branch_ids:
            parent_idx = branch_info[idx].parent
            if parent_idx is None:  # root branch
                continue
            length = current_step - branch_info[idx].birth_step
            prefix = completion_ids[idx, :-length]
            assert torch.all(prefix == completion_ids[parent_idx, :-length])
            assert completion_ids[idx, -length] != completion_ids[parent_idx, -length]

            a = completion_ids[idx, -length:]
            b = completion_ids[parent_idx, -length:]
            if torch.any((a == self.eos_id) | (b == self.eos_id)):
                continue
            model_batch.append({"prefix": prefix, "a": a, "b": b})
            meta.append((idx, length))

        if self.keep_math_core_symbols:
            model_batch = list(
                filter(
                    lambda x: self.tokenizer.decode([x["b"][0].item()]).strip() not in math_core_symbols,
                    model_batch,
                )
            )

        if not model_batch:
            return remove_set

        # Model scores in one call
        if self.filter_model is not None:
            scores = self.filter_model(**self.collator(model_batch)).scores
            for (idx, _), score in zip(meta, scores):
                if score > self.model_filter_thres:
                    remove_set.add(idx)

        # Edit distance criteria in one pass
        if self.rollout_filter_edit_dist_thres is not None:
            for (idx, length), sample in zip(meta, model_batch):
                ratio = edit_distance(sample["a"], sample["b"]) / length
                if ratio < self.rollout_filter_edit_dist_thres:
                    remove_set.add(idx)

        return remove_set


class CumulativeProbFilter(SequenceFilter):
    def __init__(
        self,
        cumulative_prob_filter_thres: float,
        cumulative_prob_filter_interval: int,
        eos_id: int,
        kt_n_gen: int,
    ):
        self.interval = cumulative_prob_filter_interval
        self.log_thres = math.log(cumulative_prob_filter_thres)
        self.eos_id = eos_id
        self.kt_n_gen = kt_n_gen

    def __call__(self, sequences: torch.Tensor, branch_info: BranchInfo, current_step: int) -> set[int]:
        if current_step % self.interval:
            return set()

        remove_set: set[int] = set()
        has_eos = (sequences == self.eos_id).any(dim=-1)
        for group in branch_info.group_by_root().values():
            if len(group) == self.kt_n_gen:
                continue  # skip those that are already full
            max_p = max(branch_info[i].accumulated_logp for i in group)
            for idx in group:
                if has_eos[idx]:
                    continue
                b = branch_info[idx]
                # always keep roots
                if b.parent is not None and b.accumulated_logp - max_p < self.log_thres * current_step:
                    remove_set.add(idx)

        return remove_set


class SequenceFilterList:
    def __init__(self, filters: Iterable[SequenceFilter]):
        self.filters = list(filters)

    def __call__(
        self, sequences: torch.Tensor, branch_info: BranchInfo, current_step: int
    ) -> tuple[set[int], dict[str, float]]:
        remove_set: set[int] = set()
        times = {}
        for filter in self.filters:
            start = time.time()
            remove_set.update(filter(sequences, branch_info, current_step))
            name = f"kt_seq_filter/{filter.__class__.__name__}"
            times[name] = time.time() - start
        return remove_set, times

    def append(self, filter: SequenceFilter):
        self.filters.append(filter)

    def __len__(self) -> int:
        return len(self.filters)

    def __getitem__(self, idx):
        return self.filters[idx]


def create_key_token_filter(
    tokenizer: PreTrainedTokenizer,
    embedding_module: Optional[torch.nn.Module] = None,
    prob_filter_abs_thres: Optional[float] = None,
    prob_filter_rel_thres: Optional[float] = None,
    similarity_filter_thres: Optional[float] = None,
    stop_word_filter: bool = False,
) -> KeyTokenFilterList:
    """
    Factory function to create a KeyTokenFilterList with specified filters.

    Args:
        tokenizer: The tokenizer to use for text processing
        embedding_module: Module for embeddings (required if using similarity filter)
        prob_filter_abs_thres: Absolute probability threshold for ProbFilter (None to disable)
        prob_filter_rel_thres: Relative probability threshold for ProbFilter (None to disable)
        similarity_filter_thres: Similarity threshold for SimilarityFilter (None to disable)
        stop_word_filter: Whether to enable StopWordFilter

    Returns:
        KeyTokenFilterList with the specified filters
    """
    filters = []

    if prob_filter_abs_thres is not None or prob_filter_rel_thres is not None:
        filters.append(ProbFilter(tokenizer, prob_filter_abs_thres, prob_filter_rel_thres))

    if stop_word_filter:
        filters.append(StopWordFilter(tokenizer))

    if similarity_filter_thres is not None:
        if embedding_module is None:
            raise ValueError("embedding_module is required when using similarity filter")
        filters.append(SimilarityFilter(tokenizer, embedding_module, similarity_filter_thres))

    if len(filters) == 0:
        raise ValueError("No filters provided")

    return KeyTokenFilterList(filters)


def create_sequence_filter(
    tokenizer: PreTrainedTokenizer,
    kt_n_gen: int,
    rollout_filter_steps: list[int],
    rollout_filter_edit_dist_thres: float | None,
    model_filter_thres: float | None,
    cumulative_prob_filter_thres: float | None,
    cumulative_prob_filter_interval: int,
    filter_model: DivJudge | None,
    keep_math_core_symbols: bool,
) -> SequenceFilterList:
    filters = []

    if rollout_filter_edit_dist_thres is not None or model_filter_thres is not None:
        rollout_filter = RolloutFilter(
            rollout_filter_steps,
            rollout_filter_edit_dist_thres,
            model_filter_thres,
            tokenizer,
            filter_model,
            keep_math_core_symbols,
        )
        filters.append(rollout_filter)

    if cumulative_prob_filter_thres is not None:
        cumulative_prob_filter = CumulativeProbFilter(
            cumulative_prob_filter_thres,
            cumulative_prob_filter_interval,
            cast(int, tokenizer.eos_token_id),
            kt_n_gen,
        )
        filters.append(cumulative_prob_filter)

    return SequenceFilterList(filters)
