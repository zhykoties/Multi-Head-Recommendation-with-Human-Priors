#!/bin/bash

#SBATCH --job-name=HLLM-NerB
#SBATCH --partition=learnai
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --output=HLLM-Nerd_%j_TinyLlama11_IN_seq24_eval8_pred4_cat6v2_NCE_2seg_081825.log
#SBATCH --error=HLLM-Nerd_%j_TinyLlama11_IN_seq24_eval8_pred4_cat6v2_NCE_2seg_081825.err

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
    --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --MAX_ITEM_LIST_LENGTH 24 \
    --optim_args.learning_rate 1e-4 \
    --checkpoint_dir /fsx/user/HLLM/saved_path \
    --loss nce \
    --num_negatives 4096 \
    --accumulate_grad 1 \
    --MAX_TEXT_LENGTH 256 \
    --dataset eb_nerd_512 \
    --save_model_note "TinyLlama11_IN_seq24_eval8_pred4_cat6v2_NCE_2seg_081825" \
    --item_pretrain_dir /fsx/user/HLLM/from_pretrained/TinyLlama-1.1B-Chat-v1.0 \
    --user_pretrain_dir /fsx/user/HLLM/from_pretrained/TinyLlama_v1.1 \
    --train_batch_size 32 \
    --gradient_checkpointing True \
    --stage 2 \
    --text_keys '[\"title\",\"subtitle\",\"topics\"]' \
    --suppress_history False \
    --medusa_num_layers 1 \
    --num_segment_head 2 \
    --num_prior_head 1 \
    --head_interaction "multiplicative" \
    --split_mode "combine" \
    --use_image_online False \
    --use_image False \
    --log_wandb True \
    --pred_len 4 \
    --eval_pred_len 8 \
    --medusa_lambda 0.99 \
    --total_iters 3000 \
    --eval_interval 300 \
    --eval_num_cats 6 \
    --neg_sample_by_cat False \
    --log_detailed_results True \
    --tag_version "v2" \
    --min_seq_len 40 \
    --outlier_user_metrics "category"
