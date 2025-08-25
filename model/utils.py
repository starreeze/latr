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

    # step-based params: in format step -> value
    # how many kt seqs are mixed in the generation
    mix_ratio_schedule: dict[int, float] = field(default_factory=lambda: {0: 1.0})

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
    # full: do sample only when branches are full; always: do sample whenever there is only one valid candidate
    sample_nk: Literal["none", "full", "always"] = "full"
    fill_return_sequences: bool = True
    # fallback to original generation for debug purpose
    # fill: fill all the branches at start and use the kt logic; native: explicit multinomial sampling
    fallback: bool = False
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
    current_step: int = 0

    def __post_init__(self):
        self.mix_ratio = self.mix_ratio_schedule.pop(0)
        print(f"step {self.current_step} mix_ratio: {self.mix_ratio}")

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
        # step based params
        self.current_step += 1
        if self.current_step in self.mix_ratio_schedule:
            self.mix_ratio = self.mix_ratio_schedule.pop(self.current_step)
        print(f"step {self.current_step} mix_ratio: {self.mix_ratio}")

        # step up/down the filter params
        if self.enable_param_scheduler:
            if suppress_ratio > self.suppress_ratio_thres:
                self._step_up()
                print("step up")
            if empty_branch_ratio > self.empty_branch_ratio_thres:
                self._step_down()
                print("step down")
        return self


@dataclass
class GenerateKeyTokenOutput(GenerateDecoderOnlyOutput):
    num_suppressed_branches: int | None = None
    scheduler: BranchParamScheduler | None = None
    suppress_ratio: float | None = None
    empty_branch_ratio: float | None = None
    num_seq: int | None = None


class MixedCache:
    def __init__(self, orig_n_samples: int, cache: DynamicCache | None = None) -> None:
        self.orig_n_samples = orig_n_samples
        self._dynamic: DynamicCache = cache if cache is not None else DynamicCache()

    def to_dynamic(self) -> DynamicCache:
        return self._dynamic

    def refresh(self, cache: DynamicCache) -> None:
        self._dynamic = cache

    def append_dup_kt_rows(self, rows: list[int]) -> None:
        if not rows:
            return
        key_cache = self._dynamic.key_cache
        value_cache = self._dynamic.value_cache
        if not key_cache:
            return
        batch_offset = self.orig_n_samples
        for i in range(len(key_cache)):
            k = key_cache[i]
            v = value_cache[i]
            idx = torch.tensor(rows, device=k.device, dtype=torch.long) + batch_offset
            k_rows = k.index_select(0, idx)
            v_rows = v.index_select(0, idx)
            key_cache[i] = torch.cat([k, k_rows], dim=0)
            value_cache[i] = torch.cat([v, v_rows], dim=0)

    def apply_full_batch_mask(self, full_mask: Tensor) -> None:
        key_cache = self._dynamic.key_cache
        value_cache = self._dynamic.value_cache
        if not key_cache:
            return
        device = key_cache[0].device
        if full_mask.device != device:
            full_mask = full_mask.to(device)
        for i in range(len(key_cache)):
            key_cache[i] = key_cache[i][full_mask]
            value_cache[i] = value_cache[i][full_mask]


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
        cache: MixedCache,
        indices_to_remove: Iterable[int],
        attention_mask: Tensor,
        orig_n_samples: int,
    ) -> tuple[Tensor, Tensor, MixedCache, Tensor]:
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

        new_attention_mask = None
        if attention_mask is not None:
            new_attention_mask = attention_mask[mask]

        full_mask = torch.cat([torch.ones(orig_n_samples, dtype=torch.bool), mask])
        new_bs = len(new_sequence) + orig_n_samples
        stop_lens[:new_bs] = stop_lens[: orig_n_samples + bs][full_mask]
        stop_lens[orig_n_samples + new_bs :] = 0

        cache.apply_full_batch_mask(full_mask)

        return new_sequence, stop_lens, cache, new_attention_mask
