from dataclasses import dataclass, field, fields
from typing import Literal, cast

import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import EosTokenCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from model.diverge import DivJudge
from model.sample_filter import (
    Branch,
    BranchInfo,
    KeyTokenFilterList,
    ModelFilter,
    RolloutFilter,
    create_key_token_filter,
)
from tools.utils import get_repeat_interleave, init_dataclass_from_dict


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
    prob_filter_thres: float | None = 0.2
    similarity_filter_thres: float | None = None
    rollout_filter_edit_dist_thres: float = 0.4
    model_filter_cand_thres: float | None = None  # the higher the more loose
    model_filter_rollout_thres: float | None = None  # the higher the more loose

    # param scheduler
    enable_param_scheduler: bool = True
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

        if self.prob_filter_thres is not None:
            self.prob_filter_thres += self.param_scheduler_step
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

        if self.prob_filter_thres is not None:
            self.prob_filter_thres -= self.param_scheduler_step
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


def _duplicate_cache_for_sequence(cache: DynamicCache | None, source_idx: int) -> DynamicCache | None:
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


def _concatenate_caches(cache_list: list[DynamicCache | None]) -> DynamicCache | None:
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
class KtModules:
    sched: BranchParamScheduler | None = None
    filter: DivJudge | None = None


class KeyTokenGenMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kt_modules = KtModules()

    def generate(
        self,
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        attention_mask: torch.Tensor | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        config: KeyTokenGenConfig | None = None,
        **kwargs,
    ) -> GenerateKeyTokenOutput:
        return generate(
            self,  # type: ignore
            self.kt_modules,
            input_ids,
            tokenizer,
            attention_mask,
            stopping_criteria,
            config,
            **kwargs,
        )


def _sample_new_sequence_with_rollout(
    key_token_filters: KeyTokenFilterList,
    probs: torch.Tensor,  # (batch, vocab)
    current_ids: torch.Tensor,  # (batch, seq_len)
    attention_mask: torch.Tensor,  # (batch, seq_len)
    cache: DynamicCache | None,
    complete_mask: torch.Tensor,  # (batch,)
    num_return_sequences: int,
    max_n_branch_per_token: int,
    sample_nk: Literal["none", "full", "always"],
    branch_info: BranchInfo,
    current_step: int,
    start: int,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor, DynamicCache | None]:
    """
    Unified batched sampling and branching across all roots.
    Maintains a single current_ids tensor and a single BranchInfo with multiple roots.
    """

    device = current_ids.device
    batch_size = current_ids.shape[0]

    # 1) Compute top-k for all rows
    topk_probs, topk_ids = probs.topk(max_n_branch_per_token, dim=-1)  # (bs, max_n_branch)

    # 2) Build variable-length prefix-trimmed sequences for valid (non-complete) rows
    valid_rows = torch.nonzero(~complete_mask, as_tuple=False).squeeze(-1)
    valid_counts = int(valid_rows.numel())
    assert valid_counts, "All sequences are already completed"

    global_valid_ids = current_ids[valid_rows, start:]
    global_valid_topk_ids = topk_ids[valid_rows]
    global_valid_topk_probs = topk_probs[valid_rows]
    global_valid_masks = key_token_filters(global_valid_ids, global_valid_topk_ids, global_valid_topk_probs)

    # 3) Build tri-state sample mask for all rows (0=keep, 1=invalid, 2=suppress)
    sample_masks = torch.ones_like(topk_ids, dtype=torch.int8)
    # assign filter results back
    sample_masks[valid_rows] = 1 - global_valid_masks.to(torch.int8)
    # first candidate is always valid for both non-complete and complete rows
    sample_masks[:, 0] = 0
    # NOTE: it is possible to put aside the finished sequences and make space for more branches

    # 4) Per-root capacity: group sequences by root and calc cap independently
    groups = branch_info.group_by_root()
    for _, group_rows in groups.items():
        active_rows = torch.tensor(group_rows, device=device)
        keep_positions = torch.where(sample_masks[active_rows] == 0)
        num_kept = int(keep_positions[0].numel())
        if num_kept > num_return_sequences:
            rows_local, cols_local = keep_positions[0], keep_positions[1]
            max_rows = int(active_rows.numel())
            combined_indices = cols_local * max_rows + rows_local
            sorted_indices = torch.argsort(combined_indices)
            to_suppress_idx = sorted_indices[num_return_sequences:]
            sup_rows_local = rows_local[to_suppress_idx]
            sup_cols_local = cols_local[to_suppress_idx]
            sup_rows_global = active_rows[sup_rows_local]
            sample_masks[sup_rows_global, sup_cols_local] = 2
            assert (sample_masks[active_rows] == 0).sum() == num_return_sequences

    assert (sample_masks == 0).sum() <= num_return_sequences * len(groups)
    for branch, sup_num in zip(branch_info, (sample_masks == 2).sum(dim=1)):
        branch.suppressed_num += int(sup_num)

    # 5) Build new tokens and genealogy
    new_tokens_on_original_branch: list[int] = []
    new_branch_seqs: list[torch.Tensor] = []
    new_branch_attn: list[torch.Tensor] = []
    new_branch_caches: list[DynamicCache | None] = []

    for row in range(batch_size):
        if complete_mask[row]:
            # keep eos for completed rows
            new_tokens_on_original_branch.append(eos_id)
            continue
        row_candidates = topk_ids[row]
        row_mask = sample_masks[row]
        valid_cands = row_candidates[row_mask == 0]
        # perform sampling for those has only one valid candidate
        if len(valid_cands) == 1 and (
            sample_nk == "always"
            or (sample_nk == "full" and branch_info.get_num_branch(row) == num_return_sequences)
        ):
            # here prob is already processed by logits_proc
            new_token = int(torch.multinomial(probs[row], num_samples=1).item())
            new_tokens_on_original_branch.append(new_token)
            continue
        assert len(valid_cands), "At least one candidate should be valid"
        for cand_idx, cand in enumerate(valid_cands.tolist()):
            if cand_idx == 0:
                new_tokens_on_original_branch.append(int(cand))
                continue
            branch_info.add_branch(row, current_step)
            # prepare new sequence and cache for the new branch
            nt_tensor = torch.tensor([[cand]], device=device, dtype=current_ids.dtype)
            new_seq = torch.cat([current_ids[row : row + 1, :], nt_tensor], dim=1)
            new_branch_seqs.append(new_seq)
            # duplicate parent attention mask and append 1
            new_attn_row = torch.cat(
                [
                    attention_mask[row : row + 1, :],
                    torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
            new_branch_attn.append(new_attn_row)
            new_branch_caches.append(_duplicate_cache_for_sequence(cache, row))
    assert len(new_tokens_on_original_branch) == len(current_ids)
    assert len(new_branch_seqs) == len(new_branch_attn) == len(new_branch_caches)
    assert len(new_branch_seqs) + len(current_ids) == (sample_masks == 0).sum()

    # 6) Append new tokens for original branches
    new_ids_col = torch.tensor(
        new_tokens_on_original_branch, dtype=current_ids.dtype, device=device
    ).unsqueeze(1)
    current_ids = torch.cat([current_ids, new_ids_col], dim=1)
    # append attention 1s for all existing rows
    ones_col = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
    attention_mask = torch.cat([attention_mask, ones_col], dim=1)

    # 7) Concatenate new sequences and caches
    if new_branch_seqs:
        current_ids = torch.cat([current_ids, *new_branch_seqs])
        attention_mask = torch.cat([attention_mask, *new_branch_attn])
        if cache is not None:
            cache = _concatenate_caches([cache, *new_branch_caches])

    assert len(current_ids) == len(attention_mask) == (sample_masks == 0).sum()
    return current_ids, attention_mask, cache


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    kt_modules: KtModules,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    attention_mask: torch.Tensor | None = None,
    stopping_criteria: StoppingCriteriaList | None = None,
    config: KeyTokenGenConfig | None = None,
    **kwargs,
) -> GenerateKeyTokenOutput:
    """
    Generate using key token tree search.
    The key token tree search is a tree search algorithm that generates a tree from the given input sequence (root sequence).
    Every time a new token is generated, the tree is split into multiple branches if this is a key token and has multiple valid candidates.
    If the number of branches exceeds num_return_sequences, the branches with lower probability are suppressed.
    """
    valid_keys = set(f.name for f in fields(KeyTokenGenConfig))
    kwargs_config = {k: v for k, v in kwargs.items() if k in valid_keys}
    if config is None:
        config = KeyTokenGenConfig(**kwargs_config)
    else:
        config.__dict__.update(kwargs_config)
    assert config.output_hidden_states is False, "output hidden states is not supported"
    assert config.num_return_sequences > 1, "num_return_sequences must be greater than 1"

    eos_id = cast(int, tokenizer.eos_token_id)
    pad_id = cast(int, tokenizer.pad_token_id)

    if kt_modules.sched is None:
        kt_modules.sched = init_dataclass_from_dict(BranchParamScheduler, config.__dict__)
    if attention_mask is None:
        attention_mask = input_ids != pad_id

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            interval = get_repeat_interleave(input_ids)
            if interval > 1 and interval != config.num_return_sequences:
                raise ValueError(
                    f"interval ({interval}) must match num_return_sequences ({config.num_return_sequences})"
                )
            input_ids = input_ids[::interval]
            attention_mask = attention_mask[::interval]
    else:
        raise ValueError(f"input_ids must be 1D or 2D, got {input_ids.ndim}D")

    # init constants
    if stopping_criteria is None:
        eos_id = cast(int, tokenizer.eos_token_id)
        stopping_criteria = StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_id)])
    key_token_filters = create_key_token_filter(
        tokenizer=tokenizer,
        embedding_module=model.get_input_embeddings(),
        prob_filter_thres=kt_modules.sched.prob_filter_thres,
        similarity_filter_thres=kt_modules.sched.similarity_filter_thres,
        stop_word_filter=config.stop_word_filter,
    )
    if config.model_filter_path is not None and kt_modules.filter is None:
        kt_modules.filter = DivJudge(config.model_filter_path)
        kt_modules.filter.to(input_ids.device)
        if kt_modules.sched.model_filter_cand_thres is not None:
            key_token_filters.append(
                ModelFilter(tokenizer, kt_modules.filter, kt_modules.sched.model_filter_cand_thres)
            )
    # in rl the model may be set to train mode in the training loop, so keep it in eval mode
    if kt_modules.filter is not None:
        kt_modules.filter.eval()
    rollout_filter = RolloutFilter(
        config.rollout_filter_steps,
        kt_modules.sched.rollout_filter_edit_dist_thres,
        kt_modules.sched.model_filter_rollout_thres,
        tokenizer,
        kt_modules.filter if kt_modules.sched.model_filter_rollout_thres is not None else None,
    )
    logits_proc = LogitsProcessorList([TemperatureLogitsWarper(temperature=config.temperature)])
    if config.top_k is not None and config.top_k > 0:
        logits_proc.append(TopKLogitsWarper(top_k=config.top_k))
    if config.top_p is not None and config.top_p < 1:
        logits_proc.append(TopPLogitsWarper(top_p=config.top_p))
    bs_roots = input_ids.shape[0]

    # Unified state
    current_ids: torch.Tensor = cast(torch.LongTensor, input_ids.clone())
    current_attention_mask: torch.Tensor = cast(torch.LongTensor, attention_mask.clone())
    # For left-padded inputs, generated tokens start at this constant index
    input_len: int = int(input_ids.shape[1])
    cache: DynamicCache | None = DynamicCache() if config.use_cache else None
    # Allocate stop_lens up to max total capacity
    max_total = bs_roots * config.num_return_sequences
    stop_lens = torch.zeros(max_total, device=input_ids.device, dtype=input_ids.dtype)
    branch_info = BranchInfo(n_roots=bs_roots)
    all_suppressed = 0

    # Generation loop (batched forward; per-root branching)
    wrapped_range = range(config.max_new_tokens)
    if config.progress_bar:
        wrapped_range = tqdm(wrapped_range, desc="Generating")
    for step in wrapped_range:
        # Forward pass
        concat_current_ids = current_ids[:, -1:] if step and config.use_cache else current_ids
        outputs = model(
            input_ids=concat_current_ids,
            attention_mask=current_attention_mask,
            past_key_values=cache,
            use_cache=config.use_cache,
            output_hidden_states=True,
            labels=None,
            return_dict=True,
            logits_to_keep=1,
        )
        if config.use_cache:
            cache = outputs.past_key_values
        logits = outputs.logits  # (bs, 1, vocab_size)
        assert logits is not None and logits.shape[1] == 1
        logits = logits[:, 0]  # (bs, vocab)

        # Unified sampling/branching
        logits_u = logits_proc(cast(torch.LongTensor, current_ids), cast(torch.FloatTensor, logits))
        probs = torch.softmax(logits_u, dim=-1)
        active_bs = current_ids.shape[0]
        complete_mask = stop_lens[:active_bs] != 0
        current_ids, current_attention_mask, cache = _sample_new_sequence_with_rollout(
            key_token_filters,
            probs,
            current_ids,
            current_attention_mask,
            cache,
            complete_mask,
            config.num_return_sequences,
            config.max_n_branch_per_token,
            config.sample_nk,
            branch_info,
            step,
            input_len,
            eos_id,
        )

        # Rollout filtering
        completion_ids = current_ids[:, input_len:]
        remove_set = rollout_filter.get_remove_indices(completion_ids, branch_info, step + 1, eos_id)
        current_ids, stop_lens, cache, new_attn = branch_info.remove(
            current_ids, stop_lens, cache, remove_set, attention_mask=current_attention_mask
        )
        if new_attn is not None:
            current_attention_mask = new_attn

        # stopping criteria
        all_done = True
        should_stop = stopping_criteria(current_ids, logits_u)  # type: ignore
        bs_now = current_ids.shape[0]
        mask = should_stop & (stop_lens[:bs_now] == 0)
        stop_lens[:bs_now][mask] = step + 1
        if not stop_lens[:bs_now].all():
            all_done = False

        if isinstance(wrapped_range, tqdm):
            all_suppressed = sum(b.suppressed_num for b in branch_info)
            # Number of branches per root (by stable root ids)
            groups = branch_info.group_by_root()
            wrapped_range.set_postfix(
                suppressed=all_suppressed, branches=[len(groups[rid]) for rid in sorted(groups.keys())]
            )
        if all_done:
            break

    # Step scheduler
    suppress_ratio = all_suppressed / (bs_roots * config.num_return_sequences * config.max_new_tokens)
    empty_branch_ratio = 1 - len(branch_info) / (bs_roots * config.num_return_sequences)
    kt_modules.sched.step(suppress_ratio, empty_branch_ratio)

    # Finalize outputs for unified batch; pad per-sequence and then global pad
    sequences_per_root: list[torch.Tensor] = []
    groups = branch_info.group_by_root()

    # Rebuild a consistent BranchInfo aligned with the reordered (and possibly duplicated) sequences
    new_branches: list[Branch] = []
    old_to_new: dict[int, int] = {}

    for _, rows in sorted(groups.items()):
        assert all(r < current_ids.shape[0] for r in rows)
        ids = current_ids[rows]
        bs_r, seq_len = ids.shape

        # align stop lengths to selected rows
        batch_stop_lens = stop_lens[rows]
        actual_seq_lens = torch.where(batch_stop_lens > 0, input_len + batch_stop_lens, seq_len)
        position_indices = torch.arange(seq_len, device=ids.device).unsqueeze(0).expand(bs_r, -1)
        should_pad = position_indices >= actual_seq_lens.unsqueeze(1)
        ids[should_pad] = pad_id

        # Mirror the reordering into new branches
        for old_idx in rows:
            old_b = branch_info[old_idx]
            new_idx = len(new_branches)
            old_to_new[old_idx] = new_idx
            new_branches.append(
                Branch(
                    parent=old_b.parent,  # temporarily old index; fix later
                    birth_step=old_b.birth_step,
                    children=set(),  # will rebuild after remap
                    suppressed_num=old_b.suppressed_num,
                    root=old_b.root,
                )
            )

        # Fill to num_return_sequences if needed (duplicate first row of this root)
        if config.fill_return_sequences and ids.shape[0] < config.num_return_sequences:
            pad_seq = ids[0].unsqueeze(0).expand(config.num_return_sequences - bs_r, -1)
            ids = torch.cat([ids, pad_seq], dim=0)

            # Duplicate the first branch meta as well
            dup_src_old_idx = rows[0]
            src_b = branch_info[dup_src_old_idx]
            for _ in range(config.num_return_sequences - bs_r):
                new_branches.append(
                    Branch(
                        parent=src_b.parent,  # temporarily old index; fix later
                        birth_step=src_b.birth_step,
                        children=set(),  # rebuilt later
                        suppressed_num=src_b.suppressed_num,
                        root=src_b.root,
                    )
                )

        sequences_per_root.append(ids)

    # Remap parent indices to new indices and rebuild children sets
    for b in new_branches:
        if b.parent is not None:
            # Map old parent index to new parent index
            b.parent = old_to_new[b.parent]
    # Rebuild children from parent links
    for idx, b in enumerate(new_branches):
        if b.parent is not None:
            parent_idx = b.parent
            new_branches[parent_idx].children.add(idx)

    sequences = torch.cat(sequences_per_root, dim=0)

    return GenerateKeyTokenOutput(
        sequences=cast(torch.LongTensor, sequences),
        num_suppressed_branches=all_suppressed,
        branch_info=BranchInfo.from_branch_list(new_branches),
        scheduler=kt_modules.sched,
        suppress_ratio=suppress_ratio,
        empty_branch_ratio=empty_branch_ratio,
        num_seq=len(branch_info),
    )
