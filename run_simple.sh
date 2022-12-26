#!/bin/bash

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO

python main.py SwinTrack Tiny --distributed_nproc_per_node 8 --distributed_do_spawn_workers --output_dir ./output --num_workers 12 --wandb_run_offline
