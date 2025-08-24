import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Literal, cast

import torch
from tqdm import tqdm
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import EosTokenCriteria, StoppingCriteriaList
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from model.diverge import DivJudge
from model.sample_filter import KeyTokenFilterList, ModelFilter, RolloutFilter, create_key_token_filter
from model.utils import (
    BranchInfo,
    BranchParamScheduler,
    GenerateKeyTokenOutput,
    KeyTokenGenConfig,
    MixedCache,
)
from tools.utils import get_repeat_interleave, init_dataclass_from_dict


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


def _init_generation_config(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: KeyTokenGenConfig | None,
    stopping_criteria: StoppingCriteriaList | None,
    kt_modules: KtModules,
    kwargs: dict,
):
    args_dict = (config.__dict__ | kwargs) if config else kwargs
    config = init_dataclass_from_dict(KeyTokenGenConfig, args_dict)
    assert config.use_cache
    assert config.output_hidden_states is False, "output hidden states is not supported"
    assert config.num_return_sequences > 1, "num_return_sequences must be greater than 1"

    eos_id = cast(int, tokenizer.eos_token_id)
    pad_id = cast(int, tokenizer.pad_token_id)

    if attention_mask is None:
        attention_mask = input_ids != pad_id

    interval = None
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
    else:
        raise ValueError(f"input_ids must be 1D or 2D, got {input_ids.ndim}D")
    if interval is not None:
        input_ids = input_ids[::interval]
        attention_mask = attention_mask[::interval]
    if config.fallback_level == "native":
        input_ids = input_ids.repeat_interleave(config.num_return_sequences, dim=0)
        attention_mask = attention_mask.repeat_interleave(config.num_return_sequences, dim=0)

    # init constants
    # for common generation
    if stopping_criteria is None:
        eos_id = cast(int, tokenizer.eos_token_id)
        stopping_criteria = StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_id)])
    logits_proc = LogitsProcessorList([TemperatureLogitsWarper(temperature=config.temperature)])
    if config.top_k is not None and config.top_k > 0:
        logits_proc.append(TopKLogitsWarper(top_k=config.top_k))
    if config.top_p is not None and config.top_p < 1:
        logits_proc.append(TopPLogitsWarper(top_p=config.top_p))
    # for key token generation
    if config.fallback_level == "native":
        print("fallback level is native; key token generation is disabled")
        key_token_filters = None
        rollout_filter = None
    else:
        if kt_modules.sched is None:
            kt_modules.sched = init_dataclass_from_dict(BranchParamScheduler, config.__dict__)
        key_token_filters = create_key_token_filter(
            tokenizer=tokenizer,
            embedding_module=model.get_input_embeddings(),
            prob_filter_abs_thres=kt_modules.sched.prob_filter_abs_thres,
            prob_filter_rel_thres=kt_modules.sched.prob_filter_rel_thres,
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
        if config.rollout_filter_edit_dist_thres is not None or config.model_filter_rollout_thres is not None:
            rollout_filter = RolloutFilter(
                config.rollout_filter_steps,
                kt_modules.sched.rollout_filter_edit_dist_thres,
                kt_modules.sched.model_filter_rollout_thres,
                tokenizer,
                kt_modules.filter if kt_modules.sched.model_filter_rollout_thres is not None else None,
            )
        else:
            rollout_filter = None

    return (
        config,
        input_ids,
        attention_mask,
        rollout_filter,
        key_token_filters,
        stopping_criteria,
        logits_proc,
    )


def _sample_new_sequence_with_rollout(
    key_token_filters: KeyTokenFilterList,
    probs: torch.Tensor,  # (batch, vocab)
    current_ids: torch.Tensor,  # (batch, seq_len)
    attention_mask: torch.Tensor,  # (batch, seq_len)
    cache: MixedCache,
    complete_mask: torch.Tensor,  # (batch,)
    num_return_sequences: int,
    max_n_branch_per_token: int,
    sample_nk: Literal["none", "full", "always"],
    branch_info: BranchInfo,
    current_step: int,
    input_len: int,
    eos_id: int,
    sync_gpus: bool,
) -> tuple[torch.Tensor, torch.Tensor, MixedCache]:
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
    if not valid_counts:
        assert sync_gpus, "All sequences are already completed"
        pad_tensor = torch.full((batch_size, 1), eos_id, device=device, dtype=current_ids.dtype)
        current_ids = torch.cat([current_ids, pad_tensor], dim=1)
        ones_col = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, ones_col], dim=1)
        return current_ids, attention_mask, cache

    global_valid_ids = current_ids[valid_rows, input_len:]
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
    dup_kt_rows: list[int] = []

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
            # Defer duplicating cache rows for these new branches
            dup_kt_rows.append(row)
    assert len(new_tokens_on_original_branch) == len(current_ids)
    assert len(new_branch_seqs) == len(new_branch_attn)
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
        cache.append_dup_kt_rows(dup_kt_rows)

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
    config, input_ids, attention_mask, rollout_filter, key_token_filters, stopping_criteria, logits_proc = (
        _init_generation_config(
            model, tokenizer, input_ids, attention_mask, config, stopping_criteria, kt_modules, kwargs
        )
    )
    eos_id = cast(int, tokenizer.eos_token_id)
    pad_id = cast(int, tokenizer.pad_token_id)

    # Unified state
    kt_n_gen = round(config.num_return_sequences * config.mix_ratio)
    assert 0 < kt_n_gen <= config.num_return_sequences
    orig_n_gen = config.num_return_sequences - kt_n_gen
    orig_n_samples = orig_n_gen * len(input_ids)
    kt_ids: torch.Tensor = cast(torch.LongTensor, input_ids.clone())
    kt_mask: torch.Tensor = cast(torch.LongTensor, attention_mask.clone())
    if orig_n_gen:
        orig_ids = input_ids.repeat_interleave(orig_n_gen, dim=0)
        orig_mask = attention_mask.repeat_interleave(orig_n_gen, dim=0)
    else:
        orig_ids = None
        orig_mask = None
    # For left-padded inputs, generated tokens start at this constant index
    input_len: int = int(input_ids.shape[1])
    cache: MixedCache = MixedCache(orig_n_samples)
    # Allocate stop_lens up to max total capacity
    bs_roots = input_ids.shape[0]
    max_total = bs_roots * config.num_return_sequences
    stop_lens = torch.zeros(max_total, device=input_ids.device, dtype=input_ids.dtype)
    branch_info = BranchInfo(n_roots=bs_roots)
    all_suppressed = 0
    times = defaultdict(float)

    if config.fallback_level == "fill":
        if config.rollout_filter_edit_dist_thres is not None or config.model_filter_rollout_thres is not None:
            print("fallback is set to fill; all rollout filters are disabled")
            config.rollout_filter_edit_dist_thres = None
            config.model_filter_rollout_thres = None
            rollout_filter = None
        children_ids = kt_ids.repeat_interleave(kt_n_gen - 1, dim=0)
        kt_ids = torch.cat([kt_ids, children_ids], dim=0)
        children_attn = attention_mask.repeat_interleave(kt_n_gen - 1, dim=0)
        kt_mask = torch.cat([kt_mask, children_attn], dim=0)
        for i, _ in product(range(bs_roots), range(kt_n_gen - 1)):
            branch_info.add_branch(i, 0)

    # Generation loop (batched forward; per-root branching)
    wrapped_range = range(config.max_new_tokens)
    if config.progress_bar:
        wrapped_range = tqdm(wrapped_range, desc="Generating")
    for step in wrapped_range:
        # Forward pass
        id_cache_start = time.time()

        full_ids = torch.cat([orig_ids, kt_ids], dim=0) if orig_ids is not None else kt_ids
        full_mask = torch.cat([orig_mask, kt_mask], dim=0) if orig_mask is not None else kt_mask
        past_key_values = cache.to_dynamic()

        forward_start = time.time()
        times["id_cache"] += forward_start - id_cache_start

        outputs = model(
            input_ids=full_ids[:, -1:] if step and config.use_cache else full_ids,
            attention_mask=full_mask,
            past_key_values=past_key_values,
            use_cache=config.use_cache,
            output_hidden_states=True,
            labels=None,
            return_dict=True,
            logits_to_keep=1,
        )

        id_cache_start = time.time()
        times["forward"] += id_cache_start - forward_start

        if config.use_cache:
            assert outputs.past_key_values is not None
            cache = MixedCache(orig_n_samples, outputs.past_key_values)

        branching_start = time.time()
        times["id_cache"] += branching_start - id_cache_start

        logits = outputs.logits  # (bs, 1, vocab_size)
        assert logits is not None and logits.shape[1] == 1
        logits = logits[:, 0]  # (bs, vocab)

        # Unified sampling/branching
        logits_u = logits_proc(cast(torch.LongTensor, full_ids), cast(torch.FloatTensor, logits))
        probs = torch.softmax(logits_u, dim=-1)

        if orig_ids is not None:
            assert orig_mask is not None
            orig_probs = probs[:orig_n_samples]
            new_tokens = torch.multinomial(orig_probs, num_samples=1)
            orig_ids = torch.cat([orig_ids, new_tokens], dim=1)
            new_attn_mask = torch.ones((orig_ids.shape[0], 1), device=orig_ids.device, dtype=orig_mask.dtype)
            orig_mask = torch.cat([orig_mask, new_attn_mask], dim=1)
        kt_probs = probs[orig_n_samples:]

        if config.fallback_level == "native":
            new_tokens = torch.multinomial(kt_probs, num_samples=1)
            kt_ids = torch.cat([kt_ids, new_tokens], dim=1)
            new_attn_mask = torch.ones((kt_ids.shape[0], 1), device=kt_ids.device, dtype=kt_mask.dtype)
            kt_mask = torch.cat([kt_mask, new_attn_mask], dim=1)
        else:
            assert key_token_filters is not None
            complete_mask = stop_lens[orig_n_samples : orig_n_samples + kt_ids.shape[0]] != 0
            kt_ids, kt_mask, cache = _sample_new_sequence_with_rollout(
                key_token_filters,
                kt_probs,
                kt_ids,
                kt_mask,
                cache,
                complete_mask,
                kt_n_gen,
                config.max_n_branch_per_token,
                config.sample_nk,
                branch_info,
                step,
                input_len,
                eos_id,
                config.sync_gpus,
            )

            # Rollout filtering
            if rollout_filter is not None:
                completion_ids = kt_ids[:, input_len:]
                remove_set = rollout_filter.get_remove_indices(completion_ids, branch_info, step + 1, eos_id)
                kt_ids, stop_lens, cache, new_attn = branch_info.remove(
                    kt_ids, stop_lens, cache, remove_set, kt_mask, orig_n_samples
                )
                if new_attn is not None:
                    kt_mask = new_attn

        # stopping criteria
        full_ids = torch.cat([orig_ids, kt_ids], dim=0) if orig_ids is not None else kt_ids
        all_done = True
        should_stop = stopping_criteria(full_ids, logits_u)  # type: ignore
        bs_now = full_ids.shape[0]
        mask = should_stop & (stop_lens[:bs_now] == 0)
        stop_lens[:bs_now][mask] = step + 1
        if not stop_lens[:bs_now].all():
            all_done = False

        if isinstance(wrapped_range, tqdm) and config.fallback_level != "native":
            all_suppressed = sum(b.suppressed_num for b in branch_info)
            wrapped_range.set_postfix(
                suppressed=all_suppressed, branches=branch_info.get_root_branches_repr()
            )

        times["branching"] += time.time() - branching_start

        if all_done and not config.sync_gpus:
            break

    print(times)

    if config.fallback_level == "native":
        return GenerateKeyTokenOutput(sequences=cast(torch.LongTensor, kt_ids))

    # Step scheduler
    assert kt_modules.sched is not None
    suppress_ratio = all_suppressed / (bs_roots * config.num_return_sequences * config.max_new_tokens)
    empty_branch_ratio = 1 - len(branch_info) / (bs_roots * config.num_return_sequences)
    kt_modules.sched.step(suppress_ratio, empty_branch_ratio)

    # Finalize outputs for unified batch
    # 1. Padding for each sequence
    seq_len = kt_ids.shape[1]
    batch_stop_lens = stop_lens[orig_n_samples : orig_n_samples + kt_ids.shape[0]]
    actual_seq_lens = torch.where(batch_stop_lens > 0, input_len + batch_stop_lens, seq_len)
    position_indices = torch.arange(seq_len, device=kt_ids.device).unsqueeze(0)
    should_pad = position_indices >= actual_seq_lens.unsqueeze(1)
    kt_ids[should_pad] = pad_id
    if orig_ids is not None:
        batch_stop_lens = stop_lens[:orig_n_samples]
        actual_seq_lens = torch.where(batch_stop_lens > 0, input_len + batch_stop_lens, seq_len)
        should_pad = position_indices >= actual_seq_lens.unsqueeze(1)
        orig_ids[should_pad] = pad_id

    # 2. Reorder sequences per root
    all_seqs: list[torch.Tensor] = []
    groups = branch_info.group_by_root()
    for root, rows in sorted(groups.items()):
        assert all(r < kt_ids.shape[0] for r in rows)
        ids = kt_ids[rows]
        bs_r, seq_len = ids.shape
        all_root_seqs = [ids]
        # Fill to num_return_sequences if needed (duplicate first row of this root)
        if config.fill_return_sequences and bs_r < kt_n_gen:
            pad_seq = ids[0].unsqueeze(0).expand(kt_n_gen - bs_r, -1)
            all_root_seqs.append(pad_seq)
        # also merge orig seqs
        if orig_ids is not None:
            all_root_seqs.append(orig_ids[root * orig_n_gen : (root + 1) * orig_n_gen])
        all_seqs.extend(all_root_seqs)
    sequences = torch.cat(all_seqs, dim=0)

    return GenerateKeyTokenOutput(
        sequences=cast(torch.LongTensor, sequences),
        num_suppressed_branches=all_suppressed,
        scheduler=kt_modules.sched,
        suppress_ratio=suppress_ratio,
        empty_branch_ratio=empty_branch_ratio,
        num_seq=len(branch_info),
    )
