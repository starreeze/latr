from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional

import nltk
import torch
from torchaudio.functional import edit_distance
from transformers.cache_utils import DynamicCache
from transformers.tokenization_utils import PreTrainedTokenizer

from model.diverge import DivCollator, DivJudge

math_core_symbols = "+-*/\\<>=()[]{}_^&%$0123456789"


@dataclass
class Branch:
    """Information about each branch for rollout filtering.

    When a branch is a root (has parent_idx == None), its ``root_id`` is a
    stable identifier that preserves the original root ordering even if
    indices are compacted during removals. Non-root branches have ``root_id``
    set to None.
    """

    parent: Optional[int]
    root: int
    birth_step: int
    children: set[int] = field(default_factory=set)
    suppressed_num: int = 0


class BranchInfo:
    """Information about the tree of branches for rollout filtering.
    ``branches[i]`` is the i-th branch (also for the i-th sequence exactly).

    Supports multiple roots in a single object. Roots are branches with
    ``parent_idx is None``. Their order is tracked via ``root_id``.
    """

    def __init__(self, n_roots: int):
        if n_roots < 0:
            raise ValueError("n_roots must be >= 0")
        self.branches = [Branch(parent=None, birth_step=0, root=i) for i in range(n_roots)]

    @staticmethod
    def from_branch_list(branches: list[Branch]) -> BranchInfo:
        res = BranchInfo(0)
        res.branches = branches
        return res

    def __repr__(self) -> str:
        head = "   idx parent  birth   root    children"
        lines = [head]
        for i, b in enumerate(self.branches):
            children = ", ".join(str(c) for c in b.children)
            parent_idx = b.parent if b.parent is not None else "None"
            lines.append(f"{i:>6} {parent_idx:>6} {b.birth_step:>6} {b.root:>6}    {children}")
        return "\n".join(lines)

    def add_branch(self, parent: int, birth_step: int):
        # Validate parent_idx
        if parent < 0 or parent >= len(self.branches):
            raise ValueError(f"Invalid parent_idx {parent}. Must be in range [0, {len(self.branches)})")

        new_branch = Branch(parent, self.branches[parent].root, birth_step)
        self.branches.append(new_branch)
        self.branches[parent].children.add(len(self.branches) - 1)

    def __getitem__(self, idx) -> Branch:
        return self.branches[idx]

    def __len__(self) -> int:
        return len(self.branches)

    def __iter__(self):
        return iter(self.branches)

    def get_branch_ids_by_birth_step(self, birth_step: int) -> list[int]:
        return [i for i, b in enumerate(self.branches) if b.birth_step == birth_step]

    def _get_recursive_remove_indices(self, indices: Iterable[int]) -> set[int]:
        indices_to_process = set(indices)
        indices_to_remove = set(indices)  # Start with the original indices

        while indices_to_process:
            idx = indices_to_process.pop()
            # Validate index
            if idx < 0 or idx >= len(self.branches):
                raise ValueError(f"Invalid index {idx}. Must be in range [0, {len(self.branches)})")

            for child in self.branches[idx].children:
                if child not in indices_to_remove:  # Avoid infinite loops
                    indices_to_process.add(child)
                    indices_to_remove.add(child)

        return indices_to_remove

    def _remove_branches(self, indices_to_remove: set[int]):
        # Validate all indices
        for idx in indices_to_remove:
            if idx < 0 or idx >= len(self.branches):
                raise ValueError(f"Invalid index {idx}. Must be in range [0, {len(self.branches)})")

        # Cannot remove any root branch
        if any(self.branches[idx].parent is None for idx in indices_to_remove):
            raise ValueError("Cannot remove a root branch")

        # Remove parent-child relationships
        for idx in indices_to_remove:
            parent = self.branches[idx].parent
            if parent is not None:
                self.branches[parent].children.remove(idx)

        # Create mapping from old indices to new indices
        new_branches: list[Branch] = []
        old_to_new_mapping: dict[int, int] = {}

        for old_idx, branch in enumerate(self.branches):
            if old_idx not in indices_to_remove:
                new_idx = len(new_branches)
                old_to_new_mapping[old_idx] = new_idx
                new_branches.append(branch)

        # Update all parent_idx and children references to use new indices
        for branch in new_branches:
            # Update parent_idx
            if branch.parent is not None:
                branch.parent = old_to_new_mapping[branch.parent]

            # Update children set
            new_children = set()
            for old_child_idx in branch.children:
                if old_child_idx in old_to_new_mapping:
                    new_children.add(old_to_new_mapping[old_child_idx])
            branch.children = new_children

        self.branches = new_branches

    def group_by_root(self) -> dict[int, list[int]]:
        """Group branch indices by their stable root_id."""
        groups: dict[int, list[int]] = {}
        for i in range(len(self.branches)):
            root_id = self.branches[i].root
            groups.setdefault(root_id, []).append(i)
        return groups

    def get_num_branch(self, id: int) -> int:
        """Return the number of branches that have the same root."""
        root_branch = self.branches[self.branches[id].root]
        return len(root_branch.children) + 1

    def remove(
        self,
        sequence: torch.Tensor,
        stop_lens: torch.Tensor,
        cache: DynamicCache | None,
        indices_to_remove: Iterable[int],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, DynamicCache | None, torch.Tensor | None]:
        """
        Remove branches and their children from the sequence and cache, updating self states.
        """
        remove_set = self._get_recursive_remove_indices(indices_to_remove)
        self._remove_branches(remove_set)

        remove_list = list(remove_set)
        bs = len(sequence)
        mask = torch.ones(bs, dtype=torch.bool)
        mask[remove_list] = False
        new_sequence = sequence[mask]
        new_bs = len(new_sequence)
        stop_lens[:new_bs] = stop_lens[:bs][mask]
        stop_lens[new_bs:] = 0

        new_attention_mask = None
        if attention_mask is not None:
            new_attention_mask = attention_mask[mask]

        new_cache = None
        if cache is not None:
            # Create a new DynamicCache instance
            new_cache = DynamicCache()
            new_cache.key_cache = []
            new_cache.value_cache = []

            # Filter each layer's cache using the same mask
            for layer_idx in range(len(cache.key_cache)):
                # Only append key cache if it exists
                if cache.key_cache[layer_idx] is not None:
                    # Filter along batch dimension: (batch, heads, seq, dim) -> (filtered_batch, heads, seq, dim)
                    filtered_key = cache.key_cache[layer_idx][mask]
                    new_cache.key_cache.append(filtered_key)

                # Only append value cache if it exists
                if cache.value_cache[layer_idx] is not None:
                    # Filter along batch dimension: (batch, heads, seq, dim) -> (filtered_batch, heads, seq, dim)
                    filtered_value = cache.value_cache[layer_idx][mask]
                    new_cache.value_cache.append(filtered_value)

        return new_sequence, stop_lens, new_cache, new_attention_mask


class KeyTokenFilter(ABC):
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

    def __init__(self, tokenizer: PreTrainedTokenizer, prob_filter_thres: float):
        super().__init__(tokenizer)
        self.prob_filter_thres = prob_filter_thres

    def __call__(
        self, sequences: torch.Tensor, candidates: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        return probs > self.prob_filter_thres


class SimilarityFilter(KeyTokenFilter):
    "only keep remaining tokens with similarity to the first token less than similarity_filter_thres"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        embedding_module: torch.nn.Module,
        similarity_filter_thres: float,
    ):
        raise NotImplementedError("SimilarityFilter is not implemented")
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
        assert candidates.ndim == 2
        mask = torch.ones_like(candidates, dtype=torch.bool, device="cpu")

        for sample_idx, candidate in enumerate(candidates):
            for token_idx, token_id in enumerate(candidate):
                # keep the first token, otherwise stop words will never be sampled out
                if token_idx == 0 or token_id == -1:
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
        first = torch.ones([bs, 1], device=scores.device, dtype=torch.bool)
        return torch.cat([first, scores < self.model_filter_thres], dim=1)


class KeyTokenFilterList:
    def __init__(self, filters: Iterable[KeyTokenFilter]):
        self.filters = list(filters)

    def __call__(
        self, sequences: torch.Tensor, candidates: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        assert candidates.ndim == 2
        mask = torch.ones_like(candidates, dtype=torch.bool)
        for filter in self.filters:
            mask = mask & filter(sequences, candidates, probs)
            if not mask[:, 1:].any():
                break
        return mask

    def append(self, filter: KeyTokenFilter):
        self.filters.append(filter)

    def __len__(self) -> int:
        return len(self.filters)

    def __getitem__(self, idx):
        return self.filters[idx]


class RolloutFilter:
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

    def get_remove_indices(
        self, completion_ids: torch.Tensor, branch_info: BranchInfo, current_step: int, eos_id: int
    ) -> set[int]:
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
            if torch.any((a == eos_id) | (b == eos_id)):
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


def create_key_token_filter(
    tokenizer: PreTrainedTokenizer,
    embedding_module: Optional[torch.nn.Module] = None,
    prob_filter_thres: Optional[float] = None,
    similarity_filter_thres: Optional[float] = None,
    stop_word_filter: bool = False,
) -> KeyTokenFilterList:
    """
    Factory function to create a KeyTokenFilterList with specified filters.

    Args:
        tokenizer: The tokenizer to use for text processing
        embedding_module: Module for embeddings (required if using similarity filter)
        prob_filter_thres: Probability threshold for ProbFilter (None to disable)
        similarity_filter_thres: Similarity threshold for SimilarityFilter (None to disable)
        stop_word_filter: Whether to enable StopWordFilter

    Returns:
        KeyTokenFilterList with the specified filters
    """
    filters = []

    if prob_filter_thres is not None:
        filters.append(ProbFilter(tokenizer, prob_filter_thres))

    if stop_word_filter:
        filters.append(StopWordFilter(tokenizer))

    if similarity_filter_thres is not None:
        if embedding_module is None:
            raise ValueError("embedding_module is required when using similarity filter")
        filters.append(SimilarityFilter(tokenizer, embedding_module, similarity_filter_thres))

    if len(filters) == 0:
        raise ValueError("No filters provided")

    return KeyTokenFilterList(filters)
