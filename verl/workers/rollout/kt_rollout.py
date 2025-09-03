# modified from verl.workers.rollout.hf_rollout.HFRollout
import copy
from typing import cast

import torch
import torch.distributed as dist
import yaml
from tensordict import TensorDict
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.distributed.tensor import DTensor  # type: ignore
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from vllm import LLM, SamplingParams

from model.generation import KeyTokenGenConfig, KtModules, generate
from tools.utils import init_dataclass_from_dict
from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.model import convert_weight_keys
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

kt_conf_path = "/dev/shm/kt/config.yaml"


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class KTRollout(BaseRollout):
    def __init__(self, module: nn.Module, path: str, config):
        assert not config.calculate_log_probs

        super().__init__()
        self.config = config
        self.module = module

        conf_dict = yaml.safe_load(open(kt_conf_path))
        self.kt_config = init_dataclass_from_dict(KeyTokenGenConfig, conf_dict)
        self.kt_config.sync_gpus = True
        # self.kt_config.progress_bar = False
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.kt_modules = KtModules()

        self.pad_token_id = cast(int, self.tokenizer.pad_token_id)
        self.eos_token_id = cast(int, self.tokenizer.eos_token_id)

        if config.get("kt_mixed_engine", True):
            self.kt_config.return_on_full = True
            self.vllm_engine = LLM(
                model=path,
                enable_sleep_mode=True,
                tensor_parallel_size=config.get("tensor_parallel_size", 1),
                distributed_executor_backend="external_launcher",
                dtype=config.dtype,
                enforce_eager=config.enforce_eager,
                gpu_memory_utilization=config.gpu_memory_utilization,
                disable_custom_all_reduce=True,
                skip_tokenizer_init=False,
                disable_log_stats=config.disable_log_stats,
                enable_chunked_prefill=config.enable_chunked_prefill,
                enable_prefix_caching=True,
                trust_remote_code=True,
                seed=config.get("seed", 0),
            )
            self.vllm_model = (
                self.vllm_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model  # type: ignore
            )
            self.vllm_engine.sleep(level=2)
        else:
            self.vllm_engine = None

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if self.vllm_engine is not None:
            rank = dist.get_rank()
            torch.cuda.empty_cache()
            self.vllm_engine.wake_up()

            params = convert_weight_keys(
                self.module.state_dict(),
                cast(PreTrainedModel, getattr(self.module, "_fsdp_wrapped_module", self.module)),
            )
            loaded_params = self.vllm_model.load_weights(
                (name, (param.to("cuda").full_tensor() if isinstance(param, DTensor) else param))  # type: ignore
                for name, param in params.items()
            )
            print(
                f"rank {rank} vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}"
            )

        batch_size = prompts.batch.batch_size[0]
        if prompts.meta_info.get("validate", False):
            mbs = self.config.val_kwargs.get("micro_batch_size", batch_size)
        else:
            mbs = self.config.get("micro_batch_size", batch_size)
        num_chunks = max((batch_size + mbs - 1) // mbs, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)

        output = [self._generate_minibatch(p) for p in batch_prompts]
        if self.vllm_engine is not None:
            self.vllm_engine.sleep(level=2)
        output = DataProto.concat(output)
        self.module.train()
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        self.module.eval()
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        batch_size, prompt_length = idx.shape
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]
        is_validate = prompts.meta_info.get("validate", False)

        config = copy.copy(self.kt_config)
        if is_validate:
            config.fallback = True
            config.num_return_sequences = self.config.val_kwargs.n
            for k in ["temperature", "top_k", "top_p"]:
                setattr(config, k, getattr(self.config.val_kwargs, k))
        else:
            config.num_return_sequences = self.config.n
            for k in ["temperature", "top_k", "top_p"]:
                setattr(config, k, getattr(self.config, k))
        config.max_new_tokens = self.config.response_length

        rank = dist.get_rank()
        with autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = generate(
                self.module,  # type: ignore
                self.kt_modules,
                idx,
                self.tokenizer,
                attention_mask,
                config=config,
            )

        assert self.kt_modules.sched is not None
        if not is_validate:
            self.kt_modules.sched.step(
                prompts.meta_info["global_steps"], output.suppress_ratio, output.empty_branch_ratio
            )

        kt_seqs = output.sequences
        generated_batch_size = kt_seqs.size(0)  # bs * num_return_sequences
        prompt = kt_seqs[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = kt_seqs[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        if self.vllm_engine is None:
            assert response_length == self.config.response_length

        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        if is_validate:
            breakpoint()

        response_position_ids = position_ids[:, -1:] + delta_position_id
        output_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # since sync_gpus is True, the response_length should be the same as
        # self.config.response_length for completed generations
        if response_length == self.config.response_length:
            response_attention_mask = get_response_mask(
                response_id=response, eos_token=self.eos_token_id, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
            batch = TensorDict(
                {
                    "prompts": prompt,
                    "responses": response,
                    "input_ids": kt_seqs,
                    "attention_mask": attention_mask,
                    "position_ids": output_position_ids,
                },
                batch_size=generated_batch_size,
            )
            return DataProto(batch=batch)

        assert self.vllm_engine is not None
        torch.cuda.empty_cache()

        target_vllm_response_length = self.config.response_length - response_length
        params = SamplingParams(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=target_vllm_response_length,
            detokenize=False,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
        )
        vllm_inputs = [{"prompt_token_ids": _pre_process_inputs(self.pad_token_id, s)} for s in kt_seqs]
        print(f"rank {rank} start vllm generation")
        outputs = self.vllm_engine.generate(prompts=vllm_inputs, sampling_params=params, use_tqdm=False)  # type: ignore

        vllm_response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response_ids = output.outputs[sample_id].token_ids
                vllm_response.append(response_ids)

        vllm_response = pad_2d_list_to_length(
            vllm_response, self.pad_token_id, max_length=target_vllm_response_length
        ).to(idx.device)

        final_seqs = torch.cat([kt_seqs, vllm_response], dim=-1)
        response = final_seqs[:, prompt_length:]
        assert response.size(1) == self.config.response_length

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        output_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=prompts.meta_info["eos_token_id"], dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": final_seqs,
                "attention_mask": attention_mask,
                "position_ids": output_position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch)
