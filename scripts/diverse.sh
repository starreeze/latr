#!/bin/bash

cleanup() {
    echo "Caught SIGINT, killing all background processes..."
    # Kill all jobs in the current shell
    kill $(jobs -p) 2>/dev/null
    exit 1
}

# Set up the trap
trap cleanup SIGINT SIGTERM

total_range=1024
num_gpus=8
step=$((total_range / num_gpus))

for ((i=0; i<num_gpus; i++)); do
    start=$((i * step))
    end=$(((i + 1) * step))
    CUDA_VISIBLE_DEVICES=$i python -m tools.diverse --start_pos $start --end_pos $end $@ &
done

wait