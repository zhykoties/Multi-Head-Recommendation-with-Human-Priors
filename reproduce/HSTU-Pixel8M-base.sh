#!/bin/bash

# --- Environment Setup for Single-GPU Execution ---
# Set the master address to the local machine
export MASTER_ADDR="localhost"
# Dynamically find a free port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Hardcode the node and process counts for a single GPU
export NNODES=1
export NPROC_PER_NODE=1
export NODE_RANK=0 # Rank is always 0 in a single-process run

# (Optional) Specify which GPU to use. '0' for the first GPU, '1' for the second, etc.
export CUDA_VISIBLE_DEVICES=0

echo "Running on a single node and single GPU."
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

# --- Main Program Execution ---

# Navigate to the code directory
cd code

# Make the TORCHRUN script executable before calling it
chmod +x ../TORCHRUN

# Launch the Python script with updated parameters
python main.py \
    --sh_script \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --rdzv_id "101" \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --config_file IDNet/hstu-size4.yaml overall/ID.yaml IDNet/hstu.yaml \
    --MAX_ITEM_LIST_LENGTH 50 \
    --optim_args.learning_rate 1e-4 \
    --checkpoint_dir /fsx/user/saved_path \
    --data_path /fsx/user/dataset \
    --text_path /fsx/user/information \
    --image_dir /fsx/user/images \
    --tag_path /fsx/user/tags \
    --loss nce \
    --num_negatives 8192 \
    --accumulate_grad 1 \
    --dataset Pixel8M \
    --save_model_note "size4_seq50_eval8_pred1_cat8v2_NCE" \
    --load_checkpoint_name HSTU-Pixel8M-size2_seq50_eval8_pred1_cat8v2_NCE_081825.pth \
    --train_batch_size 64 \
    --eval_batch_size 256 \
    --gradient_checkpointing True \
    --stage 2 \
    --medusa_num_layers 0 \
    --num_segment_head 1 \
    --num_prior_head 1 \
    --head_interaction "multiplicative" \
    --split_mode "combine" \
    --use_image False \
    --log_wandb True \
    --pred_len 1 \
    --eval_pred_len 8 \
    --medusa_lambda 0.99 \
    --total_iters 30000 \
    --eval_interval 3000 \
    --eval_num_cats 8 \
    --log_detailed_results False \
    --save_for_eval False \
    --tag_version "v2" \
    --min_seq_len 50 \
    --outlier_user_metrics "category" \
    --val_only False