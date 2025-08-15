# copyed from verl.workers.rollout.hf_rollout.HFRollout
import contextlib
import copy
from typing import cast

import torch
import torch.distributed
import yaml
from tensordict import TensorDict
from torch import nn
from transformers import AutoTokenizer

from model.generation import KeyTokenGenConfig, KtModules, generate
from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask
from verl.workers.rollout.base import BaseRollout

kt_conf_path = "/dev/shm/kt/config.yaml"


class KTRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

        conf_dict = yaml.safe_load(open(kt_conf_path))
        self.kt_config = KeyTokenGenConfig(**conf_dict["kt"])
        self.kt_modules = KtModules()
        self.tokenizer = AutoTokenizer.from_pretrained(conf_dict["path"])
        self.pad_token_id = cast(int, self.tokenizer.pad_token_id)
        self.eos_token_id = cast(int, self.tokenizer.eos_token_id)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
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

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        # if isinstance(self.module, FSDP):
        #     # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
        #     param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = generate(
                self.module,  # type: ignore
                self.kt_modules,
                idx,
                self.tokenizer,
                attention_mask,
                config=config,
            )
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

        self.module.train()
        return DataProto(batch=batch)
