from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Literal

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerateDecoderOnlyOutput


@dataclass
class GenConfig:
    max_new_tokens: int = 4096
    temperature: float = 0.2
    top_k: int | None = None
    top_p: float = 1
    do_sample: bool = True
    use_cache: bool = True
    compile_generation: bool = False
    cache_implementation: Literal["dynamic", "static"] = "static"
    progress_bar: bool = True
    num_return_sequences: int = 1


@dataclass
class BranchDynamicParam:
    # dynamic filter params
    prob_filter_abs_thres: float | None = 0.2
    prob_filter_rel_thres: float | None = 0.2
    similarity_filter_thres: float | None = None
    rollout_filter_edit_dist_thres: float | None = None
    model_filter_cand_thres: float | None = None  # the higher the more loose
    model_filter_rollout_thres: float | None = None  # the higher the more loose

    # param scheduler
    enable_param_scheduler: bool = False
    param_scheduler_step: float = 0.01  # strength for each step
    max_n_step: int = 10  # max number of steps allowed for dynamic tuning
    # step up (strict) the scheduler when the suppress ratio (n_suppressed / seq_len for each branch)
    # is greater than this threshold
    suppress_ratio_thres: float = 0.1
    # step down (loose) the scheduler when the empty branch ratio (n_empty / (n_gen * batch))
    # is greater than this threshold
    empty_branch_ratio_thres: float = 0.05


@dataclass
class KeyTokenGenConfigMixin(BranchDynamicParam):
    output_hidden_states: bool = False
    max_n_branch_per_token: int = 2
    mix_ratio: float = 1  # how many kt seqs are mixed in the generation
    # full: do sample only when branches are full; always: do sample whenever there is only one valid candidate
    sample_nk: Literal["none", "full", "always"] = "full"
    fill_return_sequences: bool = True
    # fallback to original generation for debug purpose
    # fill: fill all the branches at start and use the kt logic; native: explicit multinomial sampling
    fallback_level: Literal["none", "fill", "native"] = "none"
    sync_gpus: bool = False

    # filters
    stop_word_filter: bool = False
    model_filter_path: str | None = None
    rollout_filter_steps: list[int] = field(default_factory=lambda: [30, 50])


@dataclass
class KeyTokenGenConfig(KeyTokenGenConfigMixin, GenConfig):
    pass


@dataclass
class BranchParamScheduler(BranchDynamicParam):
    adjusted_steps: int = 0

    def _step_up(self):  # more strict
        if self.adjusted_steps >= self.max_n_step:
            return
        self.adjusted_steps += 1

        if self.prob_filter_abs_thres is not None:
            self.prob_filter_abs_thres += self.param_scheduler_step
        if self.prob_filter_rel_thres is not None:
            self.prob_filter_rel_thres -= self.param_scheduler_step
        if self.similarity_filter_thres is not None:
            self.similarity_filter_thres -= self.param_scheduler_step
        if self.rollout_filter_edit_dist_thres is not None:
            self.rollout_filter_edit_dist_thres += self.param_scheduler_step
        if self.model_filter_cand_thres is not None:
            self.model_filter_cand_thres -= self.param_scheduler_step
        if self.model_filter_rollout_thres is not None:
            self.model_filter_rollout_thres -= self.param_scheduler_step

    def _step_down(self):  # more loose
        if self.adjusted_steps <= -self.max_n_step:
            return
        self.adjusted_steps -= 1

        if self.prob_filter_abs_thres is not None:
            self.prob_filter_abs_thres -= self.param_scheduler_step
        if self.prob_filter_rel_thres is not None:
            self.prob_filter_rel_thres += self.param_scheduler_step
        if self.similarity_filter_thres is not None:
            self.similarity_filter_thres += self.param_scheduler_step
        if self.rollout_filter_edit_dist_thres is not None:
            self.rollout_filter_edit_dist_thres -= self.param_scheduler_step
        if self.model_filter_cand_thres is not None:
            self.model_filter_cand_thres += self.param_scheduler_step
        if self.model_filter_rollout_thres is not None:
            self.model_filter_rollout_thres += self.param_scheduler_step

    def step(self, suppress_ratio: float, empty_branch_ratio: float):
        if self.enable_param_scheduler:
            if suppress_ratio > self.suppress_ratio_thres:
                self._step_up()
            if empty_branch_ratio > self.empty_branch_ratio_thres:
                self._step_down()
        return self


@dataclass
class GenerateKeyTokenOutput(GenerateDecoderOnlyOutput):
    num_suppressed_branches: int | None = None
    branch_info: BranchInfo | None = None
    scheduler: BranchParamScheduler | None = None
    suppress_ratio: float | None = None
    empty_branch_ratio: float | None = None
    num_seq: int | None = None


def duplicate_cache_for_sequence(cache: DynamicCache | None, source_idx: int) -> DynamicCache | None:
    """
    Helper function to duplicate a single sequence's cache state from a DynamicCache.

    Args:
        cache: The cache to duplicate from, or None
        source_idx: The index of the source sequence in the batch

    Returns:
        A cache containing only the specified sequence's state, or None if input was None
    """
    if cache is None:
        return None

    # Create a new DynamicCache instance
    new_cache = DynamicCache()
    new_cache.key_cache = []
    new_cache.value_cache = []

    # Extract the cache for the specific sequence
    for layer_idx in range(len(cache.key_cache)):
        if cache.key_cache[layer_idx] is not None:
            # Extract only the source sequence: (batch, heads, seq, dim) -> (1, heads, seq, dim)
            key_tensor = cache.key_cache[layer_idx][source_idx : source_idx + 1]
            new_cache.key_cache.append(key_tensor)

        if cache.value_cache[layer_idx] is not None:
            # Extract only the source sequence: (batch, heads, seq, dim) -> (1, heads, seq, dim)
            value_tensor = cache.value_cache[layer_idx][source_idx : source_idx + 1]
            new_cache.value_cache.append(value_tensor)

    return new_cache


def concatenate_caches(cache_list: list[DynamicCache | None]) -> DynamicCache | None:
    """
    Helper function to concatenate multiple caches along the batch dimension.

    Args:
        cache_list: list of caches to concatenate

    Returns:
        A single cache with concatenated batch dimensions, or None if all inputs are None
    """
    # Filter out None caches
    valid_caches = [cache for cache in cache_list if cache is not None]

    if not valid_caches:
        return None

    assert len(valid_caches) == len(cache_list)

    # Create a new DynamicCache instance
    new_cache = DynamicCache()
    new_cache.key_cache = []
    new_cache.value_cache = []

    # Concatenate caches layer by layer
    for layer_idx in range(len(valid_caches[0].key_cache)):
        # Concatenate key caches for this layer
        key_tensors = [
            cache.key_cache[layer_idx] for cache in valid_caches if cache.key_cache[layer_idx] is not None
        ]
        if key_tensors:
            concatenated_key = torch.cat(key_tensors, dim=0)  # Concatenate along batch dimension
            new_cache.key_cache.append(concatenated_key)

        # Concatenate value caches for this layer
        value_tensors = [
            cache.value_cache[layer_idx] for cache in valid_caches if cache.value_cache[layer_idx] is not None
        ]
        if value_tensors:
            concatenated_value = torch.cat(value_tensors, dim=0)  # Concatenate along batch dimension
            new_cache.value_cache.append(concatenated_value)

    return new_cache


@dataclass
class Branch:
    """Information about each branch for rollout filtering.

    When a branch is a root (has parent_idx == None), its ``root_id`` is a
    stable identifier that preserves the original root ordering even if
    indices are compacted during removals. Non-root branches have ``root_id``
    set to None.
    """

    parent: int | None
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

    def get_root_branches_repr(self, max_n_display=8) -> str:
        groups = self.group_by_root()
        nums = [len(groups[rid]) for rid in sorted(groups.keys())]
        if len(nums) > max_n_display:
            nums = [f"{n}x{c}" for n, c in Counter(nums).most_common(max_n_display)]
        return "[" + ",".join(str(n) for n in nums) + "]"

    def remove(
        self,
        sequence: Tensor,
        stop_lens: Tensor,
        cache: DynamicCache | None,
        indices_to_remove: Iterable[int],
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, DynamicCache | None, Tensor | None]:
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
