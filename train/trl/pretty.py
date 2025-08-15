from trl import __version__

if __version__ != "0.17.0":
    raise ValueError("PrettyGRPOTrainer requires trl version 0.17.0")

import random
from dataclasses import dataclass
from typing import Optional, Union, cast

import torch
import wandb
from datasets import Dataset, IterableDataset
from peft.config import PeftConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer, RewardFunc

from tools.utils import camel_to_snake


@dataclass
class PrettyGRPOConfig(GRPOConfig):
    wandb_log_completions: bool = False
    print_table_completion_ratio: int = 3


# this method is copied from trl.trainer.utils.print_prompt_completions_sample
# and adjusted layout to fit the long completion text
def print_prompt_completions_sample(
    prompts: list[str],
    completions: list[str],
    rewards: dict[str, list[float]],
    step: int,
    num_samples: int | None = None,
    completion_ratio: int = 3,
    reward_width: int = 8,
) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns with better width allocation
    table.add_column("Prompt", style="bright_yellow", ratio=1)
    table.add_column("Completion", style="bright_green", ratio=completion_ratio)
    table.add_column("Reward", style="bold cyan", width=reward_width, overflow="fold")

    # Some basic input validation
    if num_samples is not None:
        if num_samples >= len(prompts):
            num_samples = None
        elif num_samples <= 0:
            return

    # Subsample data if num_samples is specified
    if num_samples is not None:
        indices = random.sample(range(len(prompts)), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]
        rewards = {key: [val[i] for i in indices] for key, val in rewards.items()}

    for i in range(len(prompts)):
        rewards_strs = []
        for key in rewards.keys():
            value_str = f"{rewards[key][i]:.2f}"
            if key.endswith("Reward"):
                key = key[: -len("Reward")]
            # Convert key from "CorrectReward" to "correct_reward"
            snake_case_key = camel_to_snake(key, split=" ")
            # Create Rich Text with soft breaking enabled at underscores
            key_text = Text(snake_case_key, overflow="fold", no_wrap=False)
            key_text.stylize("dim")
            value_text = Text(value_str, style="bold", no_wrap=False)

            # Combine key and value with newline
            combined_text = Text()
            combined_text.append(key_text)
            combined_text.append("\n")
            combined_text.append(value_text)

            rewards_strs.append(combined_text)

        # Combine all reward texts with double newlines
        final_rewards_text = Text()
        for j, reward_text in enumerate(rewards_strs):
            if j > 0:
                final_rewards_text.append("\n\n")
            final_rewards_text.append(reward_text)

        table.add_row(Text(prompts[i]), Text(completions[i].replace("\n\n", "\n")), final_rewards_text)
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


class PrettyGRPOTrainer(GRPOTrainer):
    """
    A trainer that prints the completions and rewards in a prettier format.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: PrettyGRPOConfig,
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
        self.wandb_log_completions = args.wandb_log_completions
        self.args = cast(PrettyGRPOConfig, args)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super(GRPOTrainer, self).log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            print_prompt_completions_sample(
                self._textual_logs["prompt"],
                self._textual_logs["completion"],
                self._textual_logs["rewards"],
                self.state.global_step,
                self.num_completions_to_print,
                self.args.print_table_completion_ratio,
            )

            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
                and self.wandb_log_completions
            ):
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})
