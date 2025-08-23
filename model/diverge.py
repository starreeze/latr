"""Divergence judgement model on key token generation"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.generic import ModelOutput


class DivCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, dtype=torch.bfloat16):
        self.tokenizer = tokenizer
        self.dtype = dtype

    def process_tensors(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor | None]:
        "prefix, a, b, label -> input_ids, attention_mask, labels"

        # Create formatted sequences: prefix + " ## " + a + " ## " + b
        sep_tokens = self.tokenizer.encode(" ## ", add_special_tokens=False)
        sep_tensor = torch.tensor(sep_tokens)

        input_ids, labels = [], []
        for sample in batch:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, seq_a, seq_b = (
                    torch.tensor(sample["prefix"]),
                    torch.tensor(sample["a"]),
                    torch.tensor(sample["b"]),
                )
            if seq_a.ndim == 0:
                seq_a = seq_a.unsqueeze(0)
            if seq_b.ndim == 0:
                seq_b = seq_b.unsqueeze(0)
            # Concatenate: prefix + " ## " + a + " ## " + b
            combined = torch.cat([p, sep_tensor.to(p.device), seq_a, sep_tensor.to(p.device), seq_b])
            input_ids.append(combined)
            labels.append(sample.get("label", None))

        pad_id = cast(int, self.tokenizer.pad_token_id)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id, padding_side="left")
        labels = torch.tensor(labels, dtype=self.dtype) if labels[0] is not None else None

        # Create attention mask
        attention_mask = torch.zeros_like(input_ids)
        for i, seq in enumerate(input_ids):
            seq_len = len(seq)
            # For left padding, attention mask is 1 for the rightmost seq_len tokens
            attention_mask[i, -seq_len:] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def process_texts(self, batch: list[dict[str, str]]) -> dict[str, Tensor | None]:
        texts = [
            (
                f"<prefix>{sample['prefix']}</prefix> "
                f"<candidate_1>{sample['a']}</candidate_1> "
                f"<candidate_2>{sample['b']}</candidate_2>"
            )
            .replace("<think>", "")
            .replace("</think>", "")
            for sample in batch
        ]
        if batch[0].get("label", None) is not None:
            labels = torch.tensor([sample.get("label", None) for sample in batch], dtype=self.dtype)
        else:
            labels = None
        input_ids = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").input_ids
        attention_mask = input_ids != self.tokenizer.pad_token_id
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __call__(self, batch: list[dict]) -> dict[str, Tensor | None]:
        if isinstance(batch[0]["prefix"], str):
            return self.process_texts(batch)
        else:
            return self.process_tensors(batch)


@dataclass
class DivOutput(ModelOutput):
    scores: Tensor
    loss: Tensor | None = None


class DivJudge(nn.Module):
    """
    Divergence judgement model based on Qwen3-embedding.

    Takes tokenized input formatted as:
    [common prefix] ## [seq 1] ## [seq 2]

    Outputs probability that the sequences lead to different final answers.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Embedding-0.6B",
        torch_dtype=torch.bfloat16,
        max_prefix_len: int = 100,
        attn_implementation: str = "flash_attention_2",
    ):
        """
        Args:
            model_path: path to the model. If it is a backbone model, the regressor will be randomly initialized.
            Otherwise, the regressor will be loaded from the model.
        """
        super().__init__()

        is_trained = os.path.exists(os.path.join(model_path, "divergence_config.json"))
        if is_trained:
            config = json.load(open(os.path.join(model_path, "divergence_config.json")))
            self.max_prefix_len = config.get("max_prefix_len", 100)
            if self.max_prefix_len != max_prefix_len:
                raise ValueError(
                    f"max_prefix_len in config file {self.max_prefix_len} does not match {max_prefix_len}"
                )
            backbone_path = os.path.join(model_path, "backbone")
            self.backbone = AutoModel.from_pretrained(
                backbone_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation
            )
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_path, padding_side="left")
        else:
            self.max_prefix_len = max_prefix_len
            self.backbone = AutoModel.from_pretrained(
                model_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.regressor = nn.Sequential(
            nn.LayerNorm(self.backbone.config.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(self.backbone.config.hidden_size, 1, dtype=torch_dtype),
        )
        if is_trained:
            regressor_path = os.path.join(model_path, "regressor.pt")
            self.regressor.load_state_dict(torch.load(regressor_path, map_location="cpu"))

    def save_pretrained(self, save_path: str):
        """Save the complete model (backbone + regressor) to a directory."""
        os.makedirs(save_path, exist_ok=True)

        # Save backbone model
        backbone_path = os.path.join(save_path, "backbone")
        self.backbone.save_pretrained(backbone_path)
        self.tokenizer.save_pretrained(backbone_path)

        # Save regressor weights
        regressor_path = os.path.join(save_path, "regressor.pt")
        torch.save(self.regressor.state_dict(), regressor_path)

        # Save config to indicate this is a divergence model
        config_path = os.path.join(save_path, "divergence_config.json")

        with open(config_path, "w") as f:
            json.dump({"model_type": "divergence_judgement", "max_prefix_len": self.max_prefix_len}, f)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor | None = None, reduction: str = "mean"
    ) -> DivOutput:
        """
        Forward pass.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
            attention_mask: Tensor of shape [batch_size, seq_len]
            labels: Tensor of shape [batch_size]

        Returns:
            Divergence probabilities [batch_size]
            Loss (if labels are provided)
        """
        input_ids = input_ids[:, -self.max_prefix_len :]
        attention_mask = attention_mask[:, -self.max_prefix_len :]
        input_ids[input_ids >= self.tokenizer.vocab_size] = self.tokenizer.pad_token_id
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        final_embeddings = outputs.last_hidden_state[:, -1]

        # Get probability through linear regressor + sigmoid
        logits = self.regressor(final_embeddings)
        probabilities = torch.sigmoid(logits).squeeze(-1)

        # Compute L1 loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(probabilities, labels, reduction=reduction)

        return DivOutput(scores=probabilities, loss=loss)
