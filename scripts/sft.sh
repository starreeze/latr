#!/bin/bash

FIXED_COMMAND_BASE=(
    accelerate launch
    --config_file configs/zero2.yaml
    --num_processes 4
    -m train.sft
)

OVERRIDABLE_ARGS_CONFIG=(
    "model|m|/home/nfs04/model/Qwen3-1.7B-Base|value"
    "dataset_name|d|countdown|value"
    "latent_type|t|explicit|value"
    "run_name|n|sft-qwen3-base-countdown|value"
    "num_train_epochs|e|1|value"
    "learning_rate|lr|1e-5|value"
    "per_device_train_batch_size|tbs|8|value"
    "per_device_eval_batch_size|ebs|8|value"
    "gradient_accumulation_steps|gas|1|value"
    "save_strategy||no|value"
)

source scripts/executor.sh
