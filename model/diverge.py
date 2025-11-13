"""The embedding-based divergence judgement model on key token generation"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class DivCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        "prefix, a, b -> input_ids, attention_mask"
        input_texts = []
        for sample in batch:
            prefix_text = self.tokenizer.decode(sample["prefix"])
            # find the last period and remove the prefix before it
            prefix_text = prefix_text[prefix_text.rfind(". ") + 2 :]
            input_texts.extend(prefix_text + self.tokenizer.decode(sample[name]) for name in ["a", "b"])
        encoding = self.tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left").to(
            batch[0]["prefix"].device
        )
        return {"input_ids": encoding.input_ids, "attention_mask": encoding.attention_mask}


class DivJudge(nn.Module):
    """
    Divergence judgement model based on Qwen3-embedding.

    Outputs probability that the sequences lead to different final answers.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Embedding-0.6B",
        torch_dtype=torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
    ):
        super().__init__()
        self.backbone: PreTrainedModel = AutoModel.from_pretrained(
            model_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation
        )

    @torch.no_grad()
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Tensor of shape [batch_size * 2, seq_len] (interleaved input_ids of a and b)
            attention_mask: Tensor of shape [batch_size * 2, seq_len] (interleaved attention_mask of a and b)

        Returns:
            Divergence probabilities [batch_size] (probability that the sequences lead to different final answers)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, -1]
        a_embeddings = embeddings[::2]
        b_embeddings = embeddings[1::2]
        return 1 - (1 + torch.cosine_similarity(a_embeddings, b_embeddings, dim=-1)) / 2
