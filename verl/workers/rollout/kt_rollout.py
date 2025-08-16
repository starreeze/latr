# modified from verl.workers.rollout.hf_rollout.HFRollout
import copy
from typing import cast

import torch
import torch.distributed
import yaml
from tensordict import TensorDict

from args import ModelArgs
from model.generation import KeyTokenGenConfig, KeyTokenGenMixin
from model.loader import load
from tools.utils import init_dataclass_from_dict
from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask
from verl.workers.rollout.base import BaseRollout

kt_conf_path = "/dev/shm/kt/config.yaml"


class KTRollout(BaseRollout):
    def __init__(self, model_path: str, config):
        super().__init__()
        self.config = config

        conf_dict = yaml.safe_load(open(kt_conf_path))
        self.offload = conf_dict.get("offload_to_cpu", False)
        self.kt_config = init_dataclass_from_dict(KeyTokenGenConfig, conf_dict)

        _args = ModelArgs(
            model=model_path,
            local_files_only=conf_dict.get("local_files_only", False),
            force_key_token_model=conf_dict["model_arch"],
            dtype="bfloat16",
            attention_implementation=conf_dict.get("attention_implementation", "flash_attention_2"),
        )
        self.model, self.tokenizer = load(_args)
        self.model.eval()

        self.pad_token_id = cast(int, self.tokenizer.pad_token_id)
        self.eos_token_id = cast(int, self.tokenizer.eos_token_id)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.model.to(device=prompts.batch.device)  # type: ignore
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        if self.offload:
            self.model.to(device="cpu")  # type: ignore
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]
        is_validate = prompts.meta_info.get("validate", False)

        config = copy.copy(self.kt_config)
        if is_validate:
            config.num_return_sequences = 1
        else:
            config.num_return_sequences = self.config.n

        assert isinstance(self.model, KeyTokenGenMixin)
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.model.generate(idx, self.tokenizer, attention_mask, config=config)
        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(
                size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype
            )
            delta_tokens = self.pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=self.eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        return DataProto(batch=batch)
