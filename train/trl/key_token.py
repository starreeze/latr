import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union, cast

import torch
from accelerate.utils import gather, gather_object
from datasets import Dataset, IterableDataset
from peft.config import PeftConfig
from torch import nn
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import GRPOTrainer, RewardFunc, nanstd

from model.generation import GenerateKeyTokenOutput, KeyTokenGenConfigMixin, KeyTokenGenMixin
from tools.utils import get_repeat_interleave
from train.trl.pretty import PrettyGRPOConfig, PrettyGRPOTrainer


@dataclass
class KeyTokenGRPOConfig(KeyTokenGenConfigMixin, PrettyGRPOConfig):
    kt_eval_strategy: Literal["kt", "original", "shrink"] = "shrink"

    def __post_init__(self):
        super().__post_init__()
        assert not self.use_vllm, "vLLM is not supported for key token grpo"
        assert self.num_generations is not None, "num_generations must be specified for key token grpo"


class KeyTokenGRPOTrainer(PrettyGRPOTrainer):
    def __init__(
        self,
        model: Union[str, KeyTokenGenMixin],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: KeyTokenGRPOConfig,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        super().__init__(
            model,
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config,
        )
        self.args = cast(KeyTokenGRPOConfig, self.args)
        generation_keys = KeyTokenGenConfigMixin.__dataclass_fields__.keys()
        self.kt_gen_args = {k: getattr(args, k) for k in generation_keys} | {
            "progress_bar": True,
            "num_return_sequences": args.num_generations,
        }
        self.common_gen_args = {
            "do_sample": True,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "use_cache": True,
            "cache_implementation": args.cache_implementation,
            "max_new_tokens": args.max_completion_length,
        }
        self.processing_class = cast(PreTrainedTokenizerBase, self.processing_class)

    def _get_generate_results(self, input_ids, attention_mask):
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if self.control.should_evaluate and self.args.kt_eval_strategy != "kt":
                # for eval, we use the original model and only eval on different sequences
                # (ignore num_generations) to save time
                if self.args.kt_eval_strategy == "shrink":
                    interval = get_repeat_interleave(input_ids)
                    input_ids = input_ids[::interval]
                    attention_mask = attention_mask[::interval]
                else:
                    interval = 1
                results = super(KeyTokenGenMixin, unwrapped_model).generate(  # type: ignore
                    input_ids, attention_mask=attention_mask, **self.common_gen_args
                )
                sched_step, suppress_ratio, empty_branch_ratio = 0, 0, 0
            else:
                # disable param scheduler for eval
                if self.control.should_evaluate:
                    kt_args = self.kt_gen_args | {"enable_param_scheduler": False}
                else:
                    kt_args = self.kt_gen_args
                outputs: GenerateKeyTokenOutput = unwrapped_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    tokenizer=self.processing_class,
                    **self.common_gen_args,
                    **kt_args,
                )
                results = outputs.sequences
                assert outputs.scheduler is not None
                sched_step = outputs.scheduler.adjusted_steps
                suppress_ratio = outputs.suppress_ratio
                empty_branch_ratio = outputs.empty_branch_ratio
                interval = 1
        return results, interval, sched_step, suppress_ratio, empty_branch_ratio

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs  # type: ignore
        ]
        prompt_inputs = self.processing_class(  # type: ignore
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)  # type: ignore
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        prompt_completion_ids, interval, sched_step, suppress_ratio, empty_branch_ratio = (
            self._get_generate_results(prompt_ids, prompt_mask)
        )
        if interval is not None and interval > 1:
            inputs = inputs[::interval]
            prompts = prompts[::interval]
            prompt_ids = prompt_ids[::interval]
            prompt_mask = prompt_mask[::interval]

        # Compute prompt length and extract completion ids
        completion_ids = prompt_completion_ids[:, prompt_ids.size(1) :]

        # Mask everything after the first EOS token
        is_eos = completion_ids == cast(int, self.processing_class.eos_token_id)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        batch_size = (
            self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""  # type: ignore
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)  # type: ignore
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]  # type: ignore
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]  # type: ignore
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)  # type: ignore

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts), (self.accelerator.process_index + 1) * len(prompts)
        )
        advantages = advantages[process_slice]

        # Log the metrics
        sched_stat = torch.tensor(
            [[sched_step, suppress_ratio, empty_branch_ratio]], device=device, dtype=torch.float32
        )
        agg_sched_stat = (
            cast(torch.Tensor, self.accelerator.gather_for_metrics(sched_stat)).mean(dim=0).tolist()
        )
        self._metrics[mode]["sched/step"].append(agg_sched_stat[0])
        self._metrics[mode]["sched/suppress_ratio"].append(agg_sched_stat[1])
        self._metrics[mode]["sched/empty_branch_ratio"].append(agg_sched_stat[2])

        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()  # type: ignore
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())  # type: ignore
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())  # type: ignore
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())  # type: ignore

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]  # type: ignore
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_mask.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_mask.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_mask.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }
