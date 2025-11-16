#!/bin/bash

#SBATCH --job-name=HSTU-NerB
#SBATCH --partition=learnai
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --output=HSTU-EBNerd_%j_size3_seq50_eval8_pred1_cluster8v3_NCE_seg0_base_090525.log
#SBATCH --error=HSTU-EBNerd_%j_size3_seq50_eval8_pred1_cluster8v3_NCE_seg0_base_090525.err

export NODE_RANK=$SLURM_NODEID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=4
echo "NODE_RANK="$NODE_RANK
echo "NNODES="$SLURM_NNODES
echo "NPROC_PER_NODE="$NPROC_PER_NODE
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
##WORLD_SIZE=$(($SLURM_NNODES *  $SLURM_NTASKS_PER_NODE))
##echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

cd code

python main.py \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --rdzv_id "$SLURM_JOB_ID" \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --config_file IDNet/hstu-size3.yaml overall/ID.yaml IDNet/hstu.yaml \
    --MAX_ITEM_LIST_LENGTH 50 \
    --optim_args.learning_rate 1e-4 \
    --checkpoint_dir /fsx/user/HLLM/saved_path \
    --loss nce \
    --num_negatives 8192 \
    --accumulate_grad 1 \
    --dataset eb_nerd_512 \
    --save_model_note "size3_seq50_eval8_pred1_cluster8v3_NCE_seg0_base_090525" \
    --train_batch_size 128 \
    --gradient_checkpointing True \
    --stage 2 \
    --text_keys '[\"title\",\"tag\",\"subtitle\",\"topics\"]' \
    --suppress_history False \
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
    --log_detailed_results True \
    --tag_version "v3" \
    --min_seq_len 100 \
    --outlier_user_metrics "category" \
    --cluster_as_tag True \
    --eval_by_cat False