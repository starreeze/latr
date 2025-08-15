#!/bin/bash

# # For the config, see the corresponding script file.
# # --- Fixed Command Base ---
# # These arguments are fixed and will always be part of the command.
# FIXED_COMMAND_BASE=(
#     accelerate launch
#     --config_file configs/zero2.yaml
#     --num_processes 4
#     -m train.sft
# )

# # --- Overridable Arguments Configuration ---
# # Define overridable arguments: "long_name|short_name|default_value|type"
# # - long_name: e.g., model_name
# # - short_name: e.g., m (optional, use empty string if none)
# # - default_value: e.g., Qwen/Qwen2.5-3B-Instruct or true/false for flags
# # - type: "value" (expects a subsequent argument) or "flag" (boolean)
# OVERRIDABLE_ARGS_CONFIG=(
#     "model||Qwen/Qwen2.5-3B-Instruct|value"
#     "run_name|n|sft|value"
#     "align_label_with_tf_mask||True|value"
#     "coco_sft_min_latent|minl|1|value"
#     "coco_sft_max_latent|maxl|10|value"
#     "num_train_epochs|e|1|value"
#     "learning_rate|lr|2e-6|value"
#     "per_device_train_batch_size|tbs|1|value"
#     "per_device_eval_batch_size|ebs|2|value"
#     "gradient_accumulation_steps|gas|4|value"
#     "save_strategy||no|value"
#     # Example of a true boolean flag
#     # "verbose|v|false|flag"
#     # "feature_x||true|flag" # A flag that is true by default
# )
# # --- End Configuration Section ---

# Check for Bash 4.0+ for associative arrays
if ((BASH_VERSINFO[0] < 4)); then
    echo "Error: Bash version 4.0 or higher is required for this script." >&2
    exit 1
fi

declare -A CURRENT_ARGS         # Stores long_name -> current value
declare -A LONG_NAME_TO_TYPE    # Stores long_name -> type (value/flag)
declare -A LONG_NAME_TO_DEFAULT # Stores long_name -> default_value
declare -A SHORT_TO_LONG_NAME   # Maps short_name -> long_name
declare -a ORDERED_LONG_NAMES   # Maintains order of long_names for command construction

# Initialize argument metadata and current values from OVERRIDABLE_ARGS_CONFIG
for item in "${OVERRIDABLE_ARGS_CONFIG[@]}"; do
    IFS='|' read -r long_name short_name default_value type <<<"$item"

    ORDERED_LONG_NAMES+=("$long_name")
    CURRENT_ARGS["$long_name"]="$default_value"
    LONG_NAME_TO_TYPE["$long_name"]="$type"
    LONG_NAME_TO_DEFAULT["$long_name"]="$default_value"

    if [[ -n "$short_name" ]]; then
        # Check for short name conflicts
        if [[ -n "${SHORT_TO_LONG_NAME[$short_name]}" ]]; then
            existing_long_name="${SHORT_TO_LONG_NAME[$short_name]}"
            echo "Error: Short name conflict detected: '-$short_name' is assigned to both '--$existing_long_name' and '--$long_name'" >&2
            echo "Please assign a different short name for one of these options in the OVERRIDABLE_ARGS_CONFIG." >&2
            exit 1
        fi
        SHORT_TO_LONG_NAME["$short_name"]="$long_name"
    fi
done

PASSTHROUGH_ARGS=()

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    arg="$1"
    consumed=0

    if [[ "$arg" == --* ]]; then # Long option
        potential_long_name="${arg#--}"
        is_negated_flag=0

        # Handle --no-flag for flags that are true by default
        if [[ "$potential_long_name" == no-* ]]; then
            actual_long_name_for_no_flag="${potential_long_name#no-}"
            if [[ -n "${LONG_NAME_TO_TYPE[$actual_long_name_for_no_flag]}" && "${LONG_NAME_TO_TYPE[$actual_long_name_for_no_flag]}" == "flag" ]]; then # && "${LONG_NAME_TO_DEFAULT[$actual_long_name_for_no_flag]}" == "true" (can be simplified, if it is a flag, --no- makes it false)
                CURRENT_ARGS["$actual_long_name_for_no_flag"]="false"
                consumed=1
                is_negated_flag=1
            fi
        fi

        if [[ $is_negated_flag -eq 0 && -n "${LONG_NAME_TO_TYPE[$potential_long_name]}" ]]; then # Regular long option
            type="${LONG_NAME_TO_TYPE[$potential_long_name]}"
            if [[ "$type" == "value" ]]; then
                if [[ $# -gt 1 ]]; then
                    CURRENT_ARGS["$potential_long_name"]="$2"
                    consumed=2
                else
                    echo "Error: Argument $arg expects a value, but none was provided." >&2
                    exit 1
                fi
            elif [[ "$type" == "flag" ]]; then
                CURRENT_ARGS["$potential_long_name"]="true"
                consumed=1
            fi # other types could be added here
        fi
    elif [[ "$arg" == -* && ${#arg} -gt 1 ]]; then # Short option (not just "-")
        potential_short_name="${arg#-}"
        if [[ -n "${SHORT_TO_LONG_NAME[$potential_short_name]}" ]]; then
            long_name="${SHORT_TO_LONG_NAME[$potential_short_name]}"
            type="${LONG_NAME_TO_TYPE[$long_name]}"
            if [[ "$type" == "value" ]]; then
                if [[ $# -gt 1 ]]; then
                    CURRENT_ARGS["$long_name"]="$2"
                    consumed=2
                else
                    echo "Error: Argument $arg (-$potential_short_name, maps to --$long_name) expects a value, but none was provided." >&2
                    exit 1
                fi
            elif [[ "$type" == "flag" ]]; then
                CURRENT_ARGS["$long_name"]="true"
                consumed=1
            fi # other types could be added here
        fi
    fi

    if [[ $consumed -gt 0 ]]; then
        shift "$consumed"
    else
        PASSTHROUGH_ARGS+=("$1")
        shift 1
    fi
done

# Construct the final command
FINAL_COMMAND=("${FIXED_COMMAND_BASE[@]}")

for long_name in "${ORDERED_LONG_NAMES[@]}"; do
    type="${LONG_NAME_TO_TYPE[$long_name]}"
    value="${CURRENT_ARGS[$long_name]}"

    if [[ "$type" == "value" ]]; then
        FINAL_COMMAND+=(--"$long_name" "$value")
    elif [[ "$type" == "flag" ]]; then
        if [[ "$value" == "true" ]]; then
            FINAL_COMMAND+=(--"$long_name")
        fi
        # If flag is false, it's omitted, which is standard.
    fi
done

# Execute the command
echo "Executing: ${FINAL_COMMAND[@]} ${PASSTHROUGH_ARGS[@]}"
"${FINAL_COMMAND[@]}" "${PASSTHROUGH_ARGS[@]}"
