import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.distributed as dist
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
from model.sample_filter import (
    KeyTokenFilterList,
    ModelFilter,
    create_key_token_filter,
    create_sequence_filter,
)
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

    if kt_modules.sched is None:
        kt_modules.sched = init_dataclass_from_dict(BranchParamScheduler, config.__dict__)
    kt_n_gen = round(config.num_return_sequences * kt_modules.sched.mix_ratio)

    # for key token generation
    if config.fallback or kt_n_gen == 0:
        print("fallback is True or kt_n_gen is 0; key token generation is disabled")
        return config, input_ids, attention_mask, None, None, None, None

    eos_id = cast(int, tokenizer.eos_token_id)
    pad_id = cast(int, tokenizer.pad_token_id)

    if attention_mask is None:
        attention_mask = input_ids != pad_id

    assert input_ids.ndim == 2, f"input_ids must be 1D or 2D, got {input_ids.ndim}D"
    interval = get_repeat_interleave(input_ids)

    if interval == 1:
        assert config.num_return_sequences > 1, "cannot perform kt generation on non-grouped inputs"
    elif config.num_return_sequences == 1:
        config.num_return_sequences = interval
    else:
        assert interval == config.num_return_sequences, "interval and n_seq do not match"

    input_ids = input_ids[::interval]
    attention_mask = attention_mask[::interval]

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
    sequence_filters = create_sequence_filter(
        tokenizer,
        kt_n_gen,
        config.rollout_filter_steps,
        kt_modules.sched.rollout_filter_edit_dist_thres,
        kt_modules.sched.model_filter_rollout_thres,
        kt_modules.sched.cumulative_prob_filter_thres,
        config.cumulative_prob_filter_interval,
        kt_modules.filter,
        config.token_filter_keep_math,
    )

    return (
        config,
        input_ids,
        attention_mask,
        sequence_filters,
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
) -> tuple[torch.Tensor, torch.Tensor, MixedCache, dict[str, float]]:
    """
    Unified batched sampling and branching across all roots.
    Maintains a single current_ids tensor and a single BranchInfo with multiple roots.
    """
    times = {}
    topk_start = time.time()

    device = current_ids.device
    batch_size = current_ids.shape[0]

    # 1) Compute top-k for all rows
    topk_probs, topk_ids = probs.topk(max_n_branch_per_token, dim=-1)  # (bs, max_n_branch)

    sample_mask_start = time.time()
    times["kt_sample/topk"] = sample_mask_start - topk_start

    # 2) Build variable-length prefix-trimmed sequences for valid (non-complete) rows
    valid_rows = torch.nonzero(~complete_mask, as_tuple=False).squeeze(-1)
    valid_counts = int(valid_rows.numel())
    if not valid_counts:
        assert sync_gpus, "All sequences are already completed"
        pad_tensor = torch.full((batch_size, 1), eos_id, device=device, dtype=current_ids.dtype)
        current_ids = torch.cat([current_ids, pad_tensor], dim=1)
        ones_col = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, ones_col], dim=1)
        return current_ids, attention_mask, cache, times

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

    capacity_assign_start = time.time()
    times["kt_sample/sample_mask"] = capacity_assign_start - sample_mask_start

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

    new_token_branch_start = time.time()
    times["kt_sample/capacity_assign"] = new_token_branch_start - capacity_assign_start

    # 5) Build new tokens and genealogy
    logps = (probs + 1e-9).log()
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
        assert len(valid_cands), "At least one candidate should be valid"
        # perform sampling for those has only one valid candidate
        if len(valid_cands) == 1 and (
            sample_nk == "always"
            or (sample_nk == "full" and branch_info.get_num_branch(row) == num_return_sequences)
        ):
            # here prob is already processed by logits_proc
            new_token = int(torch.multinomial(probs[row], num_samples=1).item())
            new_tokens_on_original_branch.append(new_token)
            branch_info[row].accumulated_logp += logps[row, new_token].item()
            continue

        # process minor candidates before the first (major) one to avoid messing up with the accumulated prob
        for cand in valid_cands.tolist()[1:]:
            new_branch = branch_info.add_branch(row, current_step)
            new_branch.accumulated_logp += logps[row, cand].item()
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
            # Defer duplicating cache rows for these new branches for efficiency
            dup_kt_rows.append(row)

        major_cand = int(valid_cands[0].item())
        new_tokens_on_original_branch.append(major_cand)
        branch_info[row].accumulated_logp += logps[row, major_cand].item()
        continue

    assert len(new_tokens_on_original_branch) == len(current_ids)
    assert len(new_branch_seqs) == len(new_branch_attn)
    assert len(new_branch_seqs) + len(current_ids) == (sample_masks == 0).sum()

    cat_id_start = time.time()
    times["kt_sample/new_token_branch"] = cat_id_start - new_token_branch_start

    # 6) Append new tokens for original branches
    new_ids_col = torch.tensor(
        new_tokens_on_original_branch, dtype=current_ids.dtype, device=device
    ).unsqueeze(1)
    current_ids = torch.cat([current_ids, new_ids_col], dim=1)
    # append attention 1s for all existing rows
    ones_col = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
    attention_mask = torch.cat([attention_mask, ones_col], dim=1)

    cat_cache_start = time.time()
    times["kt_sample/cat_id"] = cat_cache_start - cat_id_start

    # 7) Concatenate new sequences and caches
    if new_branch_seqs:
        current_ids = torch.cat([current_ids, *new_branch_seqs])
        attention_mask = torch.cat([attention_mask, *new_branch_attn])
        cache.append_dup_kt_rows(dup_kt_rows)

    times["kt_sample/cat_cache"] = time.time() - cat_cache_start

    assert len(current_ids) == len(attention_mask) == (sample_masks == 0).sum()
    return current_ids, attention_mask, cache, times


def maybe_cat_org_kt(orig: torch.Tensor | None, kt: torch.Tensor | None, dim=0) -> torch.Tensor:
    if orig is None:
        assert kt is not None
        return kt
    if kt is None:
        assert orig is not None
        return orig
    return torch.cat([orig, kt], dim=dim)


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
    # init config & constants
    config, input_ids, attention_mask, sequence_filters, key_token_filters, stopping_criteria, logits_proc = (
        _init_generation_config(
            model, tokenizer, input_ids, attention_mask, config, stopping_criteria, kt_modules, kwargs
        )
    )
    eos_id = cast(int, tokenizer.eos_token_id)
    pad_id = cast(int, tokenizer.pad_token_id)
    assert kt_modules.sched is not None
    kt_n_gen = 0 if config.fallback else round(config.num_return_sequences * kt_modules.sched.mix_ratio)
    if kt_n_gen == 0 and config.return_on_full:
        return GenerateKeyTokenOutput(sequences=cast(torch.LongTensor, input_ids))

    assert logits_proc is not None and attention_mask is not None

    orig_n_gen = config.num_return_sequences - kt_n_gen
    orig_n_samples = orig_n_gen * len(input_ids)
    input_len: int = int(input_ids.shape[1])
    bs_roots = input_ids.shape[0]
    printed_complete = False

    # states to be updated in each step
    if kt_n_gen:
        kt_ids = cast(torch.LongTensor, input_ids.clone())
        kt_mask = cast(torch.LongTensor, attention_mask.clone())
    else:
        kt_ids = kt_mask = None
    if orig_n_gen:
        orig_ids = input_ids.repeat_interleave(orig_n_gen, dim=0)
        orig_mask = attention_mask.repeat_interleave(orig_n_gen, dim=0)
    else:
        orig_ids = None
        orig_mask = None
    # For left-padded inputs, generated tokens start at this constant index
    cache: MixedCache = MixedCache(orig_n_samples)
    # Allocate stop_lens up to max total capacity
    max_total = bs_roots * config.num_return_sequences
    stop_lens = torch.zeros(max_total, device=input_ids.device, dtype=input_ids.dtype)
    branch_info = BranchInfo(n_roots=bs_roots)
    all_suppressed = 0
    times = defaultdict(float)

    # Generation loop (batched forward; per-root branching)
    wrapped_range = range(config.max_new_tokens)
    if config.progress_bar:
        wrapped_range = tqdm(wrapped_range, desc="Generating")
    for step in wrapped_range:
        forward_start = time.time()

        full_ids = maybe_cat_org_kt(orig_ids, kt_ids)
        full_mask = maybe_cat_org_kt(orig_mask, kt_mask)
        outputs = model(
            input_ids=full_ids[:, -1:] if step and config.use_cache else full_ids,
            attention_mask=full_mask,
            past_key_values=cache.to_dynamic(),
            use_cache=True,
            output_hidden_states=True,
            labels=None,
            return_dict=True,
            logits_to_keep=1,
        )

        prob_start = time.time()
        times["forward"] += prob_start - forward_start

        assert outputs.past_key_values is not None
        cache = MixedCache(orig_n_samples, outputs.past_key_values)

        logits = outputs.logits  # (bs, 1, vocab_size)
        assert logits is not None and logits.shape[1] == 1
        logits = logits[:, 0]  # (bs, vocab)

        logits_u = logits_proc(cast(torch.LongTensor, full_ids), cast(torch.FloatTensor, logits))
        probs = torch.softmax(logits_u, dim=-1)

        sample_start = time.time()
        times["prob"] += sample_start - prob_start

        if orig_ids is not None:
            assert orig_mask is not None
            orig_probs = probs[:orig_n_samples]
            new_tokens = torch.multinomial(orig_probs, num_samples=1)
            orig_ids = torch.cat([orig_ids, new_tokens], dim=1)
            new_attn_mask = torch.ones((orig_ids.shape[0], 1), device=orig_ids.device, dtype=orig_mask.dtype)
            orig_mask = torch.cat([orig_mask, new_attn_mask], dim=1)

            orig_end = time.time()
            times["orig_sample"] += orig_end - sample_start
            sample_start = orig_end

        if kt_n_gen:
            assert kt_ids is not None and kt_mask is not None
            assert key_token_filters is not None
            kt_probs = probs[orig_n_samples:]
            complete_mask = stop_lens[orig_n_samples : orig_n_samples + kt_ids.shape[0]] != 0
            kt_ids, kt_mask, cache, kt_times = _sample_new_sequence_with_rollout(
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
            for k, v in kt_times.items():
                times[k] += v

            filter_start = time.time()
            times["kt_sample"] += filter_start - sample_start

            # Rollout filtering
            if sequence_filters is not None:
                completion_ids = kt_ids[:, input_len:]
                remove_set, sf_times = sequence_filters(completion_ids, branch_info, step + 1)

                for k, v in sf_times.items():
                    times[k] += v
                update_remove_start = time.time()
                times["kt_seq_filter"] += update_remove_start - filter_start

                kt_ids, stop_lens, cache, new_attn, rm_times = branch_info.remove(
                    kt_ids, stop_lens, cache, remove_set, kt_mask, orig_n_samples
                )
                for k, v in rm_times.items():
                    times[k] += v
                if new_attn is not None:
                    kt_mask = new_attn

                times["kt_update_remove"] += time.time() - update_remove_start

        # stopping criteria
        stop_start = time.time()
        full_ids = maybe_cat_org_kt(orig_ids, kt_ids)
        should_stop = stopping_criteria(full_ids, logits_u)  # type: ignore
        bs_now = full_ids.shape[0]
        assert bs_now == orig_n_samples + (kt_ids.shape[0] if kt_ids is not None else 0)
        mask = should_stop & (stop_lens[:bs_now] == 0)
        stop_lens[:bs_now][mask] = step + 1

        # check whether all kt seqs are full and already pass the point of branch removal
        kt_stop_mask = stop_lens[orig_n_samples:bs_now] != 0
        groups = branch_info.group_by_root()
        # all completed are regarded as full
        group_eff_lens = [kt_n_gen if kt_stop_mask[group].all() else len(group) for group in groups.values()]
        thres = config.return_nb_thres_init - config.return_nb_thres_decay * step / config.max_new_tokens
        all_kt_complete = sum(group_eff_lens) >= round(kt_n_gen * bs_roots * thres)

        rank = dist.get_rank()
        if all_kt_complete:
            max_step = max(config.rollout_filter_steps)
            for branch in branch_info:
                if branch.birth_step + max_step >= step:
                    all_kt_complete = False
                    break
            if all_kt_complete and not printed_complete:
                print(f"rank {rank} prepare to stop due to all kt seqs are completed at step {step}")
                printed_complete = True

        if isinstance(wrapped_range, tqdm) and kt_n_gen:
            all_suppressed = sum(b.suppressed_num for b in branch_info)
            wrapped_range.set_postfix(
                suppressed=all_suppressed, branches=branch_info.get_root_branches_repr()
            )

        times["stop_criteria"] += time.time() - stop_start

        if stop_lens[:bs_now].all() and not config.sync_gpus:
            print("stopping due to stop criteria")
            break
        # Synchronize KT early stop across ranks so vLLM phase starts together
        if config.return_on_full:
            global_all_kt_complete = all_kt_complete
            if config.sync_gpus:
                # Use MIN reduction on {0,1} flags: stop only when all ranks are ready
                flag_tensor = torch.tensor(
                    1 if all_kt_complete else 0, device=full_ids.device, dtype=torch.int32
                )
                dist.all_reduce(flag_tensor, op=dist.ReduceOp.MIN)
                global_all_kt_complete = bool(flag_tensor.item() == 1)
            if global_all_kt_complete:
                print(f"rank {rank} stopping due to all kt seqs are full")
                break

    print(str(times))

    # Step scheduler
    suppress_ratio = all_suppressed / (bs_roots * config.num_return_sequences * config.max_new_tokens)
    empty_branch_ratio = 1 - len(branch_info) / (bs_roots * config.num_return_sequences)

    if orig_ids is not None:
        seq_len = orig_ids.shape[1]
        batch_stop_lens = stop_lens[:orig_n_samples]
        actual_seq_lens = torch.where(batch_stop_lens > 0, input_len + batch_stop_lens, seq_len)
        position_indices = torch.arange(seq_len, device=orig_ids.device).unsqueeze(0)
        should_pad = position_indices >= actual_seq_lens.unsqueeze(1)
        orig_ids[should_pad] = pad_id

    if kt_n_gen == 0:
        return GenerateKeyTokenOutput(sequences=cast(torch.LongTensor, orig_ids))

    # pad for kt_ids
    assert kt_ids is not None
    seq_len = kt_ids.shape[1]
    batch_stop_lens = stop_lens[orig_n_samples : orig_n_samples + kt_ids.shape[0]]
    actual_seq_lens = torch.where(batch_stop_lens > 0, input_len + batch_stop_lens, seq_len)
    position_indices = torch.arange(seq_len, device=kt_ids.device).unsqueeze(0)
    should_pad = position_indices >= actual_seq_lens.unsqueeze(1)
    kt_ids[should_pad] = pad_id

    # Reorder sequences per root
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
