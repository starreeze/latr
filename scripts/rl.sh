#!/bin/bash

FIXED_COMMAND_BASE=(
    python scripts/trl.py --gpu_ids 0 1 2 3 4 5 6 7
    # python -m pdb -m train.grpo --report_to none
)

OVERRIDABLE_ARGS_CONFIG=(
    "model|m|/home/xingsy/data_91/model/Qwen2.5-3B|value"
    "dataset_name|d|countdown|value"
    "template|tp|plain|value"
    "max_completion_length|l|1024|value"
    "per_device_train_batch_size|tbs|4|value"
    "per_device_eval_batch_size|ebs|256|value"
    "num_generations|n|16|value"
    "learning_rate|lr|1e-6|value"
    "gradient_accumulation_steps|gas|4|value"
    "eval_steps|es|20|value"
    "save_strategy||no|value"
    "run_name|r|qwen2.5-3b-kt-countdown|value"
    "prob_filter_thres|pt|0.1|value"
    "model_filter_path|mp|outputs/diverge|value"
)

source scripts/executor.sh
