#!/bin/bash

#SBATCH --job-name=HLLM-Pixel8M
#SBATCH --partition=learnai
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=HLLM-Pixel8M-Qwen2_img_IN_seq10_eval8_pred4_cat8v2_prior_seg2_hier_embed_min20_eval_092225.log
#SBATCH --error=HLLM-Pixel8M-Qwen2_img_IN_seq10_eval8_pred4_cat8v2_prior_seg2_hier_embed_min20_eval_092225.err

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
    --MAX_ITEM_LIST_LENGTH 10 \
    --optim_args.learning_rate 1e-4 \
    --checkpoint_dir /user/final_checkpoint/ \
    --loss prior \
    --accumulate_grad 1 \
    --MAX_TEXT_LENGTH 256 \
    --dataset Pixel8M \
    --save_model_note "Qwen2_img_IN_seq10_eval8_pred4_cat8v2_prior_seg2_hier_embed_min20_collector0729fp32_eval_092225" \
    --load_checkpoint_name HLLM-Pixel8M-Qwen2_img_IN_seq10_eval8_pred4_cat8v2_prior_seg2_hier_embed_min20_collector0729fp32_081725.pth \
    --item_pretrain_dir /user/fsx/from_pretrained/Qwen2-VL-2B-Instruct \
    --user_pretrain_dir /user/fsx/from_pretrained/Qwen2.5-1.5B \
    --train_batch_size 16 \
    --gradient_checkpointing True \
    --stage 2 \
    --medusa_num_layers 1 \
    --num_segment_head 2 \
    --num_prior_head 8 \
    --head_interaction "hierarchical" \
    --split_mode "combine" \
    --use_image_online False \
    --use_image True \
    --log_wandb True \
    --img_height 224 \
    --img_width 224 \
    --pred_len 4 \
    --eval_pred_len 8 \
    --medusa_lambda 0.99 \
    --total_iters 3000 \
    --eval_interval 3000 \
    --eval_num_cats 8 \
    --neg_sample_by_cat True \
    --neg_sample_mix_ratio 0 \
    --pos_sample_mix_ratio 0 \
    --weighted_prior_loss True \
    --log_detailed_results True \
    --tag_version "v2" \
    --outlier_user_metrics "category" \
    --min_seq_len 20 \
    --segment_embed True \
    --val_only True \
    --save_for_eval False
