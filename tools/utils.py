import dataclasses
import os
import random
import re
from types import UnionType
from typing import Any, Iterable, Literal, TypeVar, Union, get_args, get_origin, overload

import numpy as np
import torch


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For PyTorch >= 1.8+
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


T = TypeVar("T")


# Function overloads to preserve exact input-output type relationships
@overload
def to_device(batch: None, device="cuda", dtype: torch.dtype | None = None) -> None: ...


@overload
def to_device(batch: dict[str, T], device="cuda", dtype: torch.dtype | None = None) -> dict[str, T]: ...


@overload
def to_device(batch: list[T], device="cuda", dtype: torch.dtype | None = None) -> list[T]: ...


@overload
def to_device(batch: tuple[T, ...], device="cuda", dtype: torch.dtype | None = None) -> tuple[T, ...]: ...


@overload
def to_device(batch: torch.Tensor, device="cuda", dtype: torch.dtype | None = None) -> torch.Tensor: ...


@overload
def to_device(batch: np.ndarray, device="cuda", dtype: torch.dtype | None = None) -> torch.Tensor: ...


@overload
def to_device(batch: T, device="cuda", dtype: torch.dtype | None = None) -> T: ...


def to_device(batch, device="cuda", dtype: torch.dtype | None = None):
    "dtype conversion for tensors will only be applied within the same basic type (e.g. int, float)"
    if batch is None:
        return None
    if hasattr(batch, "items"):
        return {k: to_device(v, device, dtype) for k, v in batch.items()}
    if isinstance(batch, list):
        return [to_device(x, device, dtype) for x in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(x, device, dtype) for x in batch)
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
        if dtype is not None and (
            torch.is_floating_point(batch) == torch.is_floating_point(torch.tensor([], dtype=dtype))
        ):
            batch = batch.to(dtype)
        return batch
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch, device=device)
        if dtype is not None:
            batch = batch.to(dtype)
        return batch
    # warnings.warn(f"to_device() cannot handle type {type(batch)}")
    return batch


def pad_sequence_with_vector(
    sequences: list[torch.Tensor], padding_vector: torch.Tensor | int | float, left_pad: bool = True
):
    """Pads a list of variable length Tensors with a given embedding vector.

    Args:
        sequences (list[torch.Tensor]): list of sequences (Tensors). Assumes sequences are (seq_len, embed_dim).
        padding_vector (torch.Tensor): The vector to use for padding (embed_dim,).
        left_pad (bool): If True, pads on the left; otherwise pads on the right. Defaults to True.

    Returns:
        torch.Tensor: Padded sequences stacked into a single tensor (batch_size, max_len, embed_dim).
    """
    if not isinstance(padding_vector, torch.Tensor):
        padding_vector = torch.tensor(padding_vector, dtype=sequences[0].dtype, device=sequences[0].device)
    max_len = max(s.shape[0] for s in sequences)
    padded_sequences = []
    for seq in sequences:
        seq_len = seq.shape[0]
        pad_len = max_len - seq_len
        if pad_len > 0:
            padding = padding_vector.unsqueeze(0).repeat(pad_len, *([1] * padding_vector.ndim))
            if left_pad:
                padded_seq = torch.cat([padding, seq], dim=0)
            else:
                padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    return torch.stack(padded_sequences)


def find_values(
    x: torch.Tensor, value: float | int | bool | list | torch.Tensor, start_pos=0, end_pos=None, reverse=False
):
    """Find the first or last indices of value in x for each row within a specified range.

    Args:
        x (torch.Tensor): Input tensor (1D or 2D).
        value (float | int | torch.Tensor): Value to search for. If tensor, must be 1D and finds consecutive occurrences.
        start_pos (int): Starting index for the search (inclusive).
        end_pos (int | None): Ending index for the search (exclusive). If None, search to the end.
        reverse (bool): If True, find the last occurrence; otherwise, find the first.

    Returns:
        torch.Tensor: Indices of the value (end_pos if not found).
    """
    assert x.dim() in [1, 2]

    # Convert scalar values to tensor for unified handling
    if isinstance(value, (int, float, bool)):
        value = torch.tensor([value], dtype=x.dtype, device=x.device)
    elif isinstance(value, list):
        value = torch.tensor(value, dtype=x.dtype, device=x.device)
    else:
        assert isinstance(value, torch.Tensor), "Value must be a scalar or a tensor"
    assert value.dim() == 1, "Value tensor must be 1D"

    pattern_len = value.shape[0]
    orig_device = x.device
    effective_end_pos = end_pos if end_pos is not None else x.shape[-1]
    not_found_value = effective_end_pos

    if x.dim() == 1:
        x_slice = x[start_pos:effective_end_pos]

        if x_slice.shape[0] < pattern_len:
            return torch.tensor(not_found_value, device=orig_device)

        if pattern_len == 1:
            # Simple case - just find the single value
            mask = x_slice == value[0]
        else:
            # Use unfold to create sliding windows for efficient pattern matching
            windows = x_slice.unfold(0, pattern_len, 1)  # shape: (num_windows, pattern_len)
            # Compare each window with the pattern using vectorized operations
            mask = (windows == value.unsqueeze(0)).all(dim=1)

        match_indices = mask.nonzero(as_tuple=True)[0]
        if match_indices.numel() == 0:
            return torch.tensor(not_found_value, device=orig_device)
        else:
            found_index = match_indices[-1] if reverse else match_indices[0]
            return found_index + start_pos
    else:
        # 2D case
        x_slice = x[:, start_pos:effective_end_pos]

        # Handle empty slice case explicitly
        if x_slice.shape[1] == 0:
            return torch.full((x.shape[0],), not_found_value, dtype=torch.long, device=orig_device)

        if x_slice.shape[1] < pattern_len:
            return torch.full((x.shape[0],), not_found_value, dtype=torch.long, device=orig_device)

        batch_size = x_slice.shape[0]

        if pattern_len == 1:
            # Simple case - just find the single value
            mask = x_slice == value[0]
        else:
            # Create sliding windows for each row using vectorized operations
            windows = x_slice.unfold(1, pattern_len, 1)  # shape: (batch_size, num_windows, pattern_len)
            # Compare each window with the pattern
            mask = (windows == value.unsqueeze(0).unsqueeze(0)).all(dim=2)  # shape: (batch_size, num_windows)

        if reverse:
            # Find the last occurrence (rfind logic)
            if mask.shape[1] == 0:
                return torch.full((batch_size,), not_found_value, dtype=torch.long, device=orig_device)

            # For vectorized approach, flip the mask and find first occurrence
            flipped_mask = torch.flip(mask, dims=[1])
            first_in_flipped = torch.argmax(flipped_mask.int(), dim=1)
            any_match = mask.any(dim=1)
            found_indices_in_slice = torch.where(
                any_match,
                mask.shape[1] - 1 - first_in_flipped,
                torch.tensor(not_found_value - start_pos, device=orig_device),
            )
        else:
            # Find the first occurrence (find logic)
            if mask.shape[1] == 0:
                return torch.full((batch_size,), not_found_value, dtype=torch.long, device=orig_device)

            found_indices_in_slice = torch.argmax(mask.int(), dim=1)
            any_match = mask.any(dim=1)
            found_indices_in_slice = torch.where(
                any_match,
                found_indices_in_slice,
                torch.tensor(not_found_value - start_pos, device=orig_device),
            )

        # Adjust indices relative to original tensor if found, otherwise keep not_found_value
        final_indices = torch.where(
            found_indices_in_slice != (not_found_value - start_pos),
            found_indices_in_slice + start_pos,
            torch.tensor(not_found_value, device=orig_device),
        )
        return final_indices


def convert_padding_side(
    mask: torch.Tensor,
    *inputs: torch.Tensor,
    mode: Literal["l2r", "r2l"],
    pos: torch.Tensor | Iterable[int] | None = None,
) -> list[torch.Tensor]:
    """
    Convert padding side of sequences.
    mask: (batch_size, seq_len)
    inputs: tuple of (batch_size, seq_len, ...)
    """
    assert mode in ["l2r", "r2l"] and mask.dim() == 2
    if pos is None:
        if mode == "l2r":
            pos = find_values(mask, 1)
        else:
            pos = find_values(mask, 1, reverse=True) + 1
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos, device=mask.device)

    batch_size, seq_len = mask.shape

    # Create index tensor for circular shift
    # For each row i, we want indices: [(pos[i] + j) % seq_len for j in range(seq_len)]
    indices = torch.arange(seq_len, device=mask.device).unsqueeze(0).expand(batch_size, -1)
    indices = (indices + pos.unsqueeze(1)) % seq_len

    res = []
    for input_tensor in (mask, *inputs):
        # Expand indices to match the extra dimensions
        expanded_indices = indices
        for _ in range(input_tensor.dim() - 2):
            expanded_indices = expanded_indices.unsqueeze(-1)
        expanded_indices = expanded_indices.expand(*input_tensor.shape)
        converted = torch.gather(input_tensor, 1, expanded_indices)
        res.append(converted)
    return res


def init_dataclass_from_dict(class_type: type[T], args: dict, auto_cast: bool = True) -> T:
    """
    Initialize a dataclass from a dictionary.
    non-existing fields will be ignored.
    """
    fields = dataclasses.fields(class_type)  # type: ignore
    d = {}
    for field in fields:
        if field.name not in args:
            continue

        name = field.name
        value = args[name]
        if not auto_cast:
            d[name] = value
            continue

        _type = field.type
        if isinstance(_type, str):
            _type = eval(_type)
        # Handle Union types (e.g., int | None). Cast to the first argument type.
        origin = get_origin(_type)
        if origin in (Union, UnionType):
            union_args = get_args(_type)
            if len(union_args) > 0:
                _type = union_args[0]
        elif origin is not None:
            _type = origin
        if value is not None:
            if isinstance(value, str):
                try:
                    _value = eval(value)
                    if isinstance(_value, _type):
                        value = _value
                except Exception:
                    pass
            else:
                try:
                    value = _type(value)
                except Exception:
                    pass
        d[name] = value

    return class_type(**d)


def update_additive_stats(stats: dict[str, Any], new_stats: dict[str, Any]):
    "add new_stats to stats for all keys. They two must have identical structure (can be nested)"
    for k, v in new_stats.items():
        if k not in stats:
            stats[k] = v
            continue
        if isinstance(v, dict):
            update_additive_stats(stats[k], v)
        else:
            stats[k] += v


def camel_to_snake(name: str, split="_") -> str:
    """Convert CamelCase to snake_case."""
    # Insert underscore before uppercase letters that follow lowercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", rf"\1{split}\2", name)
    # Insert underscore before uppercase letters that follow lowercase letters or digits
    return re.sub("([a-z0-9])([A-Z])", rf"\1{split}\2", s1).lower()


def snake_to_camel(name: str, split="_") -> str:
    """Convert snake_case to CamelCase."""
    return "".join(word.capitalize() for word in name.split(split))


def is_same_sequence(seqs: torch.Tensor) -> bool:
    assert seqs.ndim == 2
    if seqs.size(0) == 1:
        return True
    remaining = seqs[1:]
    first = seqs[0:1].expand(remaining.shape[0], -1)
    return bool(torch.all(first == remaining))


def get_repeat_interleave(x: torch.Tensor) -> int:
    """
    shrink the interleaved tensor to the original tensor (reverse operation of torch.repeat_interleave).
    Return the interleave factor.
    If not interleaved, raise ValueError. Input tensor can be any dimension.

    Example:
        torch.repeat_interleave([1, 2, 3, 4], 2) -> [1, 1, 2, 2, 3, 3, 4, 4]
        shrink_interleave([1, 1, 2, 2, 3, 3, 4, 4]) -> 2
    """
    if x.numel() == 0:
        return 1

    first_dim_size = x.size(0)

    if first_dim_size == 1:
        return 1

    # Find the interval between first element and the next different element
    first_element = x[0]
    interval = None

    for i in range(1, first_dim_size):
        if not torch.equal(x[i], first_element):
            interval = i
            break

    # If no different element found, all elements are the same
    if interval is None:
        return first_dim_size

    # Check if this interval creates a valid interleaved pattern
    if first_dim_size % interval != 0:
        raise ValueError("Input tensor is not interleaved")

    original_size = first_dim_size // interval

    # Validate the pattern: each group of 'interval' consecutive elements should be identical
    for i in range(original_size):
        start_idx = i * interval
        end_idx = start_idx + interval
        group = x[start_idx:end_idx]

        # Check if all elements in this group are the same
        if not torch.all(group == group[0]):
            raise ValueError("Input tensor is not interleaved")

    res = x[::interval]

    # Verify that all elements in the result are different
    if res.ndim == 1:
        # 1D: use unique to detect duplicates efficiently
        if torch.unique(res).numel() != res.numel():
            raise ValueError("Input tensor is not interleaved")
    else:
        # For ND tensors, flatten features and compare rows pairwise
        rows = res.view(res.shape[0], -1)
        eq = (rows.unsqueeze(1) == rows.unsqueeze(0)).all(dim=-1)
        if torch.any(eq.fill_diagonal_(False)):
            raise ValueError("Input tensor is not interleaved")

    return interval


def suffix_match_len(a: torch.Tensor, b: torch.Tensor) -> int:
    assert a.ndim == 1 and b.ndim == 1 and a.shape[0] == b.shape[0] > 0
    la = a.tolist()
    lb = b.tolist()

    def longest_prefix_in_text(pattern: list, text: list) -> int:
        """
        Return the maximum length L such that pattern[:L] occurs as a contiguous
        subsequence in text. Runs in O(len(pattern) + len(text)).
        """
        m = len(pattern)
        if m == 0:
            return 0

        # Build prefix function (pi) for KMP
        pi = [0] * m
        j = 0
        for i in range(1, m):
            while j > 0 and pattern[i] != pattern[j]:
                j = pi[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            pi[i] = j

        # Scan text, tracking longest prefix match length observed
        q = 0
        best = 0
        for t in text:
            while q > 0 and pattern[q] != t:
                q = pi[q - 1]
            if pattern[q] == t:
                q += 1
            if q > best:
                best = q
            if q == m:
                # Allow overlaps to continue scanning
                q = pi[q - 1]
        return best

    # Suffix of a equals prefix of reversed(a); match anywhere in reversed(b)
    ra = la[::-1]
    rb = lb[::-1]
    l1 = longest_prefix_in_text(ra, rb)

    # Suffix of b equals prefix of reversed(b); match anywhere in reversed(a)
    l2 = longest_prefix_in_text(rb, ra)

    return max(l1, l2)
