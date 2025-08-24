import os
from dataclasses import dataclass, field
from typing import Literal

from model.utils import KeyTokenGenConfigMixin


@dataclass
class ModelArgs:
    """Arguments pertaining to model configuration."""

    model: str = field(default="Qwen/Qwen2.5-3B-Instruct", metadata={"help": "The model name or path."})
    local_files_only: bool | None = field(
        default=None, metadata={"help": "Whether to only use local model files."}
    )
    force_key_token_model: Literal["qwen2", "qwen3", "gpt2", "llama"] | None = field(
        default=None,
        metadata={
            "help": "The key token model name to force load. If not none, "
            "will use the key token model by replacing the generate method."
        },
    )
    dtype: str = field(default="auto", metadata={"help": "The dtype to use."})
    attention_implementation: str = field(default="flash_attention_2")


@dataclass
class DataArgs:
    """Arguments pertaining to what data to use."""

    # for common data
    dataset_name: str = field(
        default="countdown", metadata={"help": "The name of the dataset to use for training."}
    )
    eval_dataset: str = field(
        default="",
        metadata={
            "help": "The name of the dataset to use for evaluation. Default to the same as dataset_name."
        },
    )
    data_dir: str = field(
        default="dataset",
        metadata={
            "help": "The directory containing the datasets. "
            "The target dataset should be located in data_dir/dataset_name"
        },
    )
    n_train_samples: int = field(default=20000, metadata={"help": "The number of training samples."})
    n_eval_samples: int = field(default=128, metadata={"help": "The number of validating samples."})
    gsm8k_cot_content: Literal["sentence", "formula"] = field(
        default="sentence",
        metadata={
            "help": "The content of gsm8k cot. sentence: the original natural language cot; "
            "formula: the extracted formula for each step, e.g. <<1+2=3>>"
        },
    )
    template: Literal["conv", "plain"] = field(
        default="conv",
        metadata={
            "help": "The template to use for the data. "
            "conv: conversation format, e.g. {system: ..., user: ..., ...}; "
            "plain: directly concat question and answer"
        },
    )
    system_prompt_type: Literal["general", "reasoning", "none"] = field(
        default="general",
        metadata={
            "help": "The type of system prompt to use. "
            "general: use the general system prompt - you are a helpful assistant; "
            "reasoning: use the reasoning system prompt - see data/prompts.py"
            "none: no system prompt"
        },
    )

    # for sft
    tf_interval: int = field(
        default=1,
        metadata={
            "help": "The interval of teacher forcing tokens. " "Only applicable if latent_type = interleave."
        },
    )
    align_label_with_tf_mask: bool = field(
        default=True, metadata={"help": "Whether to only compute loss on teacher forcing tokens."}
    )
    latent_type: Literal["interleave", "dynamic", "enforce", "explicit"] = field(
        default="explicit",
        metadata={
            "help": "The latent strategy for SFT. "
            "interleave: supervise one latent token every tf_interval (as if it is an explicit token); "
            "dynamic: no supervision on latent tokens and let the model to determine latent length; "
            "enforce: preplan mode - forcefully add start and end according to the preplan; "
            "explicit: no latent tokens - use explicit cot."
        },
    )
    cot_type: Literal["xml", "plain", "instruct", "none"] = field(
        default="plain",
        metadata={
            "help": "The format of cot regardless of dataset and content. "
            "xml: <think>...</think><answer>...</answer>; "
            r"plain: [start_marker] reasoning... [end_marker] \boxed{answer}; "
            "instruct: similar to plain but is more detailed; "
            "none: just answer."
        },
    )
    plain_cot_start_marker: str = field(
        default="<think>", metadata={"help": "The start marker for the plain cot."}
    )
    plain_cot_end_marker: str = field(
        default="</think>", metadata={"help": "The end marker for the plain cot."}
    )
    sft_latent_supervision: Literal["none", "coco"] = field(
        default="none",
        metadata={
            "help": "The type of latent supervision to use. "
            "none: do not use supervision/compression on latent tokens; "
            "coco: compress stages into latent tokens, aligning with coconut"
        },
    )

    ## args for latent_supervision = none
    coco_sft_min_latent: int = field(
        default=1, metadata={"help": "The minimum number of latent tokens in coco SFT."}
    )
    coco_sft_max_latent: int = field(
        default=10, metadata={"help": "The maximum number of latent tokens in coco SFT."}
    )
    difficulty_transform: bool = field(
        default=False, metadata={"help": "Whether to transform the difficulty of the data."}
    )
    difficulty_transform_mean: float = field(
        default=0.5, metadata={"help": "The mean of the difficulty transform."}
    )
    difficulty_transform_std: float = field(
        default=0.25, metadata={"help": "The std of the difficulty transform."}
    )
    difficulty_normalize_factor: float = field(
        default=1, metadata={"help": "The factor of the difficulty normalization."}
    )

    # for eval
    score_file: str = field(default="scores.csv", metadata={"help": "The file to save the scores."})
    math_eval_timeout: int = field(
        default=5, metadata={"help": "The timeout for math evaluation. 0 means no timeout. "}
    )

    # for difficulty calculation and eval
    n_gpus: int = field(default=1, metadata={"help": "The number of GPUs to use to process data."})
    start_pos: int = field(default=0, metadata={"help": "The start position of the data to use."})
    end_pos: int = field(default=int(1e12), metadata={"help": "The end position of the data to use."})
    generate_batch_size: int = field(default=1, metadata={"help": "The batch size for generation."})

    # for diverge
    diverge_seqlen_k: list[int] = field(default_factory=lambda: [20, 30, 50])
    diverge_balance_label: bool = field(
        default=True, metadata={"help": "Whether to balance the label by subsampling."}
    )
    diverge_max_prefix_len: int | None = field(
        default=100, metadata={"help": "The maximum length of the prefix."}
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type: ignore
        if not self.eval_dataset:
            self.eval_dataset = self.dataset_name


@dataclass
class TrainingArgs:
    """Arguments pertaining to the training configuration."""

    output_dir: str = field(
        default="outputs",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    run_name: str = field(default="grpo", metadata={"help": "The name of the run."})
    learning_rate: float = field(
        default=1e-6, metadata={"help": "The initial learning rate for AdamW optimizer."}
    )
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer."})
    adam_beta2: float = field(default=0.99, metadata={"help": "Beta2 for AdamW optimizer."})
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay for AdamW optimizer."})
    warmup_ratio: float = field(
        default=0.1, metadata={"help": "Linear warmup ratio over the training steps."}
    )
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The learning rate scheduler type."})
    optim: str = field(default="adamw_8bit", metadata={"help": "The optimizer to use."})
    logging_steps: int = field(default=1, metadata={"help": "Log every X updates steps."})
    bf16: bool = field(default=True)
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16 (mixed) precision."})
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=2, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing."}
    )
    torch_compile: bool = field(default=False, metadata={"help": "Whether to use torch.compile."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    max_steps: int = field(
        default=-1,
        metadata={
            "help": "If > 0: set total number of training steps to perform. Override num_train_epochs."
        },
    )
    num_train_epochs: int = field(default=1)
    save_strategy: str = field(
        default="steps", metadata={"help": "The strategy to use for saving checkpoints."}
    )
    save_total_limit: int = field(default=2, metadata={"help": "The total number of checkpoints to save."})
    save_steps: int = field(default=1000, metadata={"help": "Save checkpoint every X updates steps."})
    eval_strategy: str = field(default="steps", metadata={"help": "The evaluation strategy to use."})
    eval_steps: int = field(default=500, metadata={"help": "Evaluate every X updates steps."})
    report_to: str = field(
        default="wandb",
        metadata={"help": "The list of integrations to report results and logs to (e.g., 'wandb')."},
    )
    eval_on_start: bool = field(default=False, metadata={"help": "Whether to evaluate on start."})
    seed: int = field(default=42, metadata={"help": "The seed to use for the training."})
    data_seed: int = field(default=0, metadata={"help": "The seed to use for the data."})
    ddp_timeout: int = field(default=5400, metadata={"help": "The timeout for DDP training."})
    resume_from_checkpoint: str | None = field(
        default=None, metadata={"help": "Whether to resume training from checkpoint."}
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type: ignore
        self.output_dir = os.path.join(self.output_dir, self.run_name)


@dataclass
class GenArgs:
    temperature: float = field(default=1.0, metadata={"help": "The temperature to use for sampling."})
    top_k: int | None = field(default=None, metadata={"help": "The top k to use for sampling."})
    top_p: float = field(default=1.0, metadata={"help": "The top p to use for sampling."})
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling or not."})
    use_cache: bool = field(default=True, metadata={"help": "Whether to use cache for generation."})
    compile_generation: bool = field(
        default=False, metadata={"help": "Whether to compile the forward method for generation."}
    )
    cache_implementation: Literal["dynamic", "static"] = field(
        default="dynamic", metadata={"help": "The implementation of the cache."}
    )
    progress_bar: bool = field(
        default=True, metadata={"help": "Whether to show a progress bar in generation."}
    )
    num_return_sequences: int = field(
        default=1, metadata={"help": "The number of return sequences for generation."}
    )


@dataclass
class GRPOArgs(GenArgs, KeyTokenGenConfigMixin, TrainingArgs):
    # common args
    use_vllm: bool = field(default=False, metadata={"help": "Whether to use vLLM for fast inference."})
    num_generations: int = field(
        default=4, metadata={"help": "Number of generations to sample from the model per prompt."}
    )
    max_prompt_length: int | None = field(
        default=None, metadata={"help": "Maximum length of the prompt. Useful for static cache."}
    )
    max_completion_length: int = field(
        default=200, metadata={"help": "Maximum length of the generated completion."}
    )
    disable_dropout: bool = field(default=True, metadata={"help": "Whether to disable dropout."})
    loss_type: str = field(default="bnpo", metadata={"help": "The type of loss to use."})
    scale_rewards: bool | None = field(default=None, metadata={"help": "Whether to scale rewards."})
    beta: float = field(default=0.04, metadata={"help": "KL coefficient."})
    epsilon_high: float = field(default=0.28, metadata={"help": "Upper-bound epsilon value for clipping."})
    use_liger_loss: bool = field(default=True, metadata={"help": "Whether to use liger loss."})

    # enhancement
    entropy_loss_ratio: float = field(
        default=1.0,
        metadata={
            "help": "The ratio of higher entropy tokens that will be used for loss calculation."
            "Refer to the paper 'Beyond the 80/20 Rule: High-Entropy Minority Tokens...'"
            "for more details."
        },
    )
    kt_eval_strategy: Literal["kt", "original", "shrink"] = field(
        default="shrink", metadata={"help": "The strategy to use for evaluation."}
    )

    # for logging
    log_completions: bool = field(default=True, metadata={"help": "Whether to log completions."})
    num_completions_to_print: int = field(default=1, metadata={"help": "Number of completions to print."})
    wandb_log_completions: bool = field(
        default=False, metadata={"help": "Whether to log completions to wandb."}
    )
    wandb_log_unique_prompts: bool = field(
        default=True, metadata={"help": "Whether to log unique prompts to wandb."}
    )
    print_table_completion_ratio: int = field(
        default=3,
        metadata={"help": "The ratio of table column for completion in relation to prompt to print."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.scale_rewards = (
            (self.loss_type != "dr_grpo") if self.scale_rewards is None else self.scale_rewards
        )


@dataclass
class RewardArgs:
    """Arguments pertaining to reward function configuration."""

    correctness_reward: float = field(default=1.0, metadata={"help": "Reward value for correct answers."})
    answer_format_reward: float = field(
        default=0.1, metadata={"help": "Reward value for answers that comply with required format."}
    )
    loose_format_reward: float = field(
        default=0.1, metadata={"help": "Reward value for responses matching the loose XML format."}
    )
    strict_format_reward: float = field(
        default=0.1, metadata={"help": "Reward value for responses matching the strict XML format."}
    )
    latent_len_reward: float = field(
        default=0, metadata={"help": "Reward value for responses with correct latent length."}
    )
    answer_len_reward: float = field(
        default=0, metadata={"help": "Reward value for responses with correct answer length."}
    )
    answer_no_think_reward: float = field(
        default=0, metadata={"help": "Reward value for answers without think words."}
    )
    response_think_reward: float = field(
        default=0, metadata={"help": "Reward value for responses with correct think length."}
    )


@dataclass
class GenerationArgs(GenArgs, KeyTokenGenConfigMixin):
    """Arguments pertaining to generation."""

    max_new_tokens: int = field(default=8192, metadata={"help": "The maximum number of tokens to generate."})
    max_input_tokens: int = field(
        default=1024, metadata={"help": "The maximum number of tokens to input. Useful for static cache. "}
    )
    max_think_tokens: int | None = field(
        default=None, metadata={"help": "The maximum number of tokens to think."}
    )
    think_marker: str = field(
        default="think",
        metadata={
            "help": "the start and end marker for the thinking tokens. Used for explicit generation only."
        },
    )
    start_think_marker: str = field(
        default="<think>",
        metadata={"help": "the start marker for the thinking tokens. Override think_marker if provided."},
    )
    end_think_marker: str = field(
        default="</think>",
        metadata={"help": "the end marker for the thinking tokens. Override think_marker if provided."},
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type: ignore
        if self.think_marker != "none":
            if not self.start_think_marker:
                self.start_think_marker = f"<{self.think_marker}>"
            if not self.end_think_marker:
                self.end_think_marker = f"</{self.think_marker}>"
