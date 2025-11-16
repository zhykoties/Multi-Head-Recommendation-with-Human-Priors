#!/bin/bash

#SBATCH --job-name=HSTU-Merrec
#SBATCH --partition=learnai
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=HSTU-Merrec_%j_size4_seq400_eval1_pred1_prior_gap0_1seg_mult_given_081325.log
#SBATCH --error=HSTU-Merrec_%j_size4_seq400_eval1_pred1_prior_gap0_1seg_mult_given_081325.err

export NODE_RANK=$SLURM_NODEID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=8
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
    --config_file IDNet/hstu-size4.yaml overall/ID.yaml IDNet/hstu.yaml \
    --MAX_ITEM_LIST_LENGTH 400 \
    --optim_args.learning_rate 1e-3 \
    --checkpoint_dir /fsx/user/HLLM/saved_path \
    --loss prior \
    --num_negatives 4096 \
    --accumulate_grad 1 \
    --dataset merrec_2000 \
    --save_model_note "size4_seq400_eval1_pred1_prior_gap0_1seg_mult_given_081325" \
    --train_batch_size 64 \
    --eval_batch_size 256 \
    --gradient_checkpointing True \
    --stage 2 \
    --text_keys '[\"category_name\",\"brand_name\"]' \
    --suppress_history False \
    --fix_temp True \
    --medusa_num_layers 1 \
    --num_segment_head 1 \
    --num_prior_head 6 \
    --head_interaction "multiplicative" \
    --split_mode "combine" \
    --log_wandb True \
    --pred_len 1 \
    --eval_pred_len 1 \
    --medusa_lambda 0.99 \
    --total_iters 30000 \
    --eval_interval 3000 \
    --eval_num_cats 6 \
    --neg_sample_by_cat False \
    --weighted_prior_loss True \
    --log_detailed_results True \
    --min_seq_len 400 \
    --prior_given_at_test True \
    --given_prior_len 1 \
    --outlier_user_metrics "event" \
    --sample_last_only False \
    --train_test_gap 0

