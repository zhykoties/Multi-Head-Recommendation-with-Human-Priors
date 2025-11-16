from REC.model.basemodel import BaseModel, all_gather
from REC.model.IDNet.hstu import HSTUJagged, SequentialTransductionUnitJagged, RelativeBucketedTimeAndPositionBasedBias, truncated_normal
from REC.utils.enum_type import InputType

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask.to(vec.dtype)
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)


class ComiRec(BaseModel):
    """
    Implements ComiRec based on HSTU.
    Original code: https://github.com/THUDM/ComiRec
    The number of dimensions d for embeddings is set to 64. The number of samples for sampled softmax
    loss is set to 10. The number of maximum training iterations is set
    to 1 million. The number of interest embeddings for multi-interest
    models is set to 4. We use Adam optimizer with learning rate lr = 0.001 for optimization.
    """
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super().__init__()
        self.logger = getLogger()
        self.item_num = dataload.item_num
        self._item_embedding_dim: int = config['item_embedding_size']
        self._hstu_embedding_dim: int = config['hstu_embedding_size']
        self.max_seq_length: int = config['MAX_ITEM_LIST_LENGTH']
        self.pred_len = config['pred_len']

        self.medusa_lambda = config['medusa_lambda']
        self.skip_hstu = config.get('skip_hstu', False)
        self.input_dropout = config.get('input_dropout', False)

        self._num_blocks: int = config['n_layers']
        self._num_heads: int = config['n_heads']
        self._dqk: int = config['hstu_embedding_size'] // config['n_heads']
        self._dv: int = config['hstu_embedding_size'] // config['n_heads']
        self._linear_activation: str = config['hidden_act'] if config['hidden_act'] else "silu"
        self._linear_dropout_rate: float = config['hidden_dropout_prob']
        self._attn_dropout_rate: float = config['attn_dropout_prob']
        self._enable_relative_attention_bias: bool = config['enable_relative_attention_bias'] if config['enable_relative_attention_bias'] else False
        self._linear_config = 'uvqk'
        self._normalization = 'rel_bias'
        self.position_embedding = nn.Embedding(self.max_seq_length+1, self._hstu_embedding_dim)
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._hstu_embedding_dim,
                    linear_hidden_dim=self._dv,
                    attention_dim=self._dqk,
                    normalization=self._normalization,
                    linear_config=self._linear_config,
                    linear_activation=self._linear_activation,
                    num_heads=self._num_heads,
                    # TODO: change to lambda x.
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=self.max_seq_length
                            + self.max_seq_length,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if self._enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=self._linear_dropout_rate,
                    attn_dropout_ratio=self._attn_dropout_rate,
                    concat_ua=False,
                )
                for _ in range(self._num_blocks)
            ],
            autocast_dtype=None,
        )

        self.input_dropout = nn.Dropout(self._linear_dropout_rate) if self.input_dropout else nn.Identity()
        self.interest_hidden = config.get('interest_hidden', self._hstu_embedding_dim // 2)
        self.num_interest = config.get('interest_num', 4)
        # Self-attention network to generate interest capsules
        self.attention_net = nn.Sequential(
            nn.Linear(self._hstu_embedding_dim, self.interest_hidden, bias=True),
            nn.Tanh(),
            nn.Dropout(self._linear_dropout_rate), 
            nn.Linear(self.interest_hidden, self.num_interest, bias=False)
        )

        self.item_embedding = nn.Embedding(self.item_num, self._item_embedding_dim, padding_idx=0)
        self.item_id_proj_tower = nn.Identity() if config['item_embedding_size'] == config['hstu_embedding_size'] else nn.Linear(config['item_embedding_size'], config['hstu_embedding_size'], bias=False)
        self.loss = config['loss']
        if self.loss in ['nce']:
            if config['fix_temp']:
                self.logger.info(f"Fixed logit_scale 20")
                self.register_buffer("logit_scale", torch.tensor(np.log(1 / 0.05)))
            else:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.05))
            self.nce_thres = config['nce_thres'] if config['nce_thres'] else 0.99
            self.logger.info(f"nce threshold setting to {self.nce_thres}")
        else:
            raise NotImplementedError(f"loss={self.loss} is not supported")

        horizon_discount = torch.tensor([self.medusa_lambda ** pred_idx for pred_idx in range(self.pred_len)])
        sum_discount = sum(horizon_discount)
        self.register_buffer('horizon_discount', horizon_discount / sum_discount)
        self.eval_pred_len = config['eval_pred_len']
        self.rank = torch.distributed.get_rank()

        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self.max_seq_length,
                        self.max_seq_length,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = True
        self.reset_params()

    def reset_params(self):
        for name, params in self.named_parameters():
            if ("_hstu" in name) or ("_embedding_module" in name) or ('logit_scale' in name):
                if self._verbose:
                    print(f"Skipping init for {name}")
                continue
            try:
                truncated_normal(params.data, mean=0.0, std=0.02)
                if self._verbose:
                    print(
                        f"Initialize {name} as trunc normal: {params.data.size()} params"
                    )
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def debug_str(self) -> str:
        debug_str = (
            f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def nce_loss(self, cur_embs_valid, target_pos_valid, neg_embedding_all):
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))
        logit_scale = self.logit_scale.exp()
        
        output_embs = F.normalize(cur_embs_valid, p=2, dim=-1)
        target_pos_embs = F.normalize(target_pos_valid, p=2, dim=-1)
        pos_logits = (output_embs * target_pos_embs).sum(-1, keepdim=True)

        # neg_embedding_all: [total_num_negs, dim]
        neg_embedding_all_t = neg_embedding_all.T.contiguous()
        neg_logits = output_embs @ neg_embedding_all_t
        # neg_logits.shape: torch.Size([batch_size, seq_len, total_num_negs])
        fix_logits = target_pos_embs @ neg_embedding_all_t
        neg_logits.masked_fill_(fix_logits > self.nce_thres, torch.finfo(neg_logits.dtype).min)

        logits = torch.cat([pos_logits, neg_logits], dim=-1) * logit_scale
        # logits.shape: torch.Size([batch_size, seq_len, total_num_negs + 1])
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        return logits, labels

    def log_topk_during_train(self, logits, labels):
        log_dict = {'nce_samples': (logits > torch.finfo(logits.dtype).min / 100).sum(
            dim=1).float().mean().detach()}
        for k in [1, 5, 10, 50, 100]:
            if k > logits.size(-1):
                break
            indices = logits.topk(k, dim=-1).indices
            log_dict[f'nce_top{k}_acc'] = labels.view(-1, 1).eq(indices).any(dim=-1).float().mean().detach()
        return log_dict

    def forward(self, interaction):
        items, neg_items, user_attention_mask, _ = interaction
        device = user_attention_mask.device
        user_attention_mask = user_attention_mask.bool()
        # items: [b_sz, seq_len+pred_len], neg_items: [b_sz, num_cats, num_negs]),
        # masked_index: [b_sz, seq_len+pred_len]
        pos_items_embs = self.item_id_proj_tower(self.item_embedding(items))  # [batch, seq_len+pred_len, dim]
        input_emb = pos_items_embs[:, :-self.pred_len, :]  # [batch, seq_len, dim]

        position_ids = torch.arange(input_emb.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(user_attention_mask[:, :-self.pred_len])
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_emb + position_embedding

        if self.skip_hstu:
            output_embs = self.input_dropout(input_emb)
        else:
            attention_mask = self.get_attention_mask(user_attention_mask[:, :-self.pred_len])
            output_embs = self._hstu(x=input_emb, attention_mask=attention_mask)

        model_out = defaultdict(float)
        B = output_embs.shape[0]
        big_batch = B * self.pred_len
        dtype = output_embs.dtype
        L = self.max_seq_length
        P = self.pred_len
        D = output_embs.size(-1)
        K = self.num_interest

        neg_embedding = self.item_id_proj_tower(self.item_embedding(neg_items[:, -1]))
        neg_embedding = F.normalize(neg_embedding, p=2, dim=-1)
        neg_embedding_all = all_gather(neg_embedding, sync_grads=True).reshape(-1, D)  # [total_num_negs, dim]

        # Calculate multi-interest user embeddings using self-attention
        # attention_net input shape: (batch_size, seq_len, embedding_dim)
        # attention_net output shape: (batch_size, seq_len, num_interest)
        padded_output_embs = F.pad(output_embs, (0, 0, L - 1, 0), "constant", 0)
        context_windows = padded_output_embs.unfold(dimension=1, size=L, step=1).permute(0, 1, 3, 2)
        # context_windows: (B, L, L, D)

        context_mask = user_attention_mask[:, :-self.pred_len] # (B, L)
        # Pad mask: (B, L+L-1)
        padded_mask = F.pad(context_mask, (L - 1, 0), "constant", 0)
        # Unfold: (B, L, L)
        context_windows_mask = padded_mask.unfold(dimension=1, size=L, step=1)

        attn_input = context_windows.reshape(B*L, L, D)
        # Apply attention net: (B*L, L, K)
        attn_weights_raw = self.attention_net(attn_input)
        # Transpose: (B*L, K, L)
        attn_weights = attn_weights_raw.permute(0, 2, 1)

        # Reshape and expand mask: (B*L, L) -> (B*L, K, L)
        mask_reshaped = context_windows_mask.reshape(B*L, L)
        mask_expanded = mask_reshaped.unsqueeze(1).expand(-1, K, -1)

        # Apply mask using a large negative number where mask is False.
        attention_weights = torch.where(mask_expanded, attn_weights, torch.finfo(attn_weights.dtype).min)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        # attention_weights = masked_softmax(attn_weights, mask_expanded, dim=-1)
        
        # Calculate Causal User Interest Embeddings
        # (B*L, K, L) @ (B*L, L, D) -> (B*L, K, D)
        user_interest_emb_flat = torch.matmul(attention_weights, attn_input)
        # Reshape back to (B, L, K, D)
        user_interest_emb_causal = user_interest_emb_flat.reshape(B, L, K, D)

        # ─── (2)  FUTURE TARGET WINDOWS & MASKS  (cheap views) ────────────────────────
        # pos_items_embs: (B, L + P, D)   user_attention_mask: (B, L + P)
        windows_pos = pos_items_embs.unfold(1, P, 1)[:, 1:].permute(0, 3, 1, 2)  # (B, P, L, D)
        windows_mask = user_attention_mask.unfold(1, P, 1)[:, 1:].permute(0, 2, 1)  # (B, P, L)
        base_mask = user_attention_mask[:, :L].unsqueeze(1)  # (B, 1, L)
        final_mask = base_mask & windows_mask  # (B, P, L)

        # Calculate similarity using the corresponding causal interests.
        # We align the L dimension (time step) between interests and targets.
        # user_interest_emb_causal: (B, L, K, D)
        # windows_pos: (B, P, L, D)
        # similarity result: (B, P, L, K)
        similarity = torch.einsum('blkd,bpld->bplk', user_interest_emb_causal, windows_pos)

        # Softmax over interests (dim=-1, K). 
        # Padded items are handled later by mask_flat.
        readout_atten = F.softmax(similarity, dim=-1)

        # "Hard Readout": Select the interest vector with the highest attention score.
        # best_interest_indices shape: (B, P, L)
        best_interest_indices = torch.argmax(readout_atten, dim=-1)
        
        # Create batch and dimension indices for advanced indexing.
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, P, L)
        # Time indices (L dimension) (B, P, L)
        l_indices = torch.arange(L, device=device).view(1, 1, L).expand(B, P, -1)

        # Select the final user representation: (B, P, L, D)
        cur_embs = user_interest_emb_causal[batch_indices, l_indices, best_interest_indices]

        # ─── (3)  FLATTEN  (batch × pred_idx)  AXIS  ──────────────────────────────────
        mask_flat = final_mask.reshape(big_batch, L).bool()
        cur_flat = cur_embs.reshape(big_batch, L, self._hstu_embedding_dim)[mask_flat]    # (BP, L, D)
        pos_flat = windows_pos.reshape(big_batch, L, self._hstu_embedding_dim)[mask_flat]

        # ─── (5)  ONE‑SHOT  InfoNCE ───────────────────────────────────────────────────
        logits, labels = self.nce_loss(cur_flat, pos_flat, neg_embedding_all)
        # logits: (N_tok, 1+N_neg)   where N_tok = mask_flat.sum()

        # ─── (6)  TOKEN‑LEVEL CROSS‑ENTROPY  ───────────────────────────────────────────
        tok_loss = F.cross_entropy(logits, labels, reduction='none')   # (N_tok,)

        # map every token back to its prediction offset 0 … P‑1
        pred_idx_token = (torch.arange(P, device=device).repeat(B)        # (BP ,)
                        .unsqueeze(1).expand(-1, L).reshape(-1)[mask_flat.reshape(-1)])

        # sum & count per offset in O(N) with one index_add_ each
        loss_sum_per_p = torch.zeros(self.pred_len, device=device, dtype=torch.float32)
        cnt_per_p = torch.zeros_like(loss_sum_per_p)
        loss_sum_per_p.index_add_(0, pred_idx_token, tok_loss.float())
        cnt_per_p.index_add_(0, pred_idx_token, torch.ones_like(tok_loss, dtype=torch.float32))
        mean_loss_per_p = (loss_sum_per_p / torch.clamp_min(cnt_per_p, 1.0)).to(dtype=dtype)

        # ─── (7)  λ–SCHEDULE & TOTAL LOSS  ────────────────────────────────────────────
        lam_vec = self.horizon_discount.to(dtype=dtype)
        per_pred_loss = lam_vec * mean_loss_per_p  # (P ,)
        model_out["loss"] += per_pred_loss.sum()

        # ─── (9)  OPTIONAL TOP‑k METRICS  (only pred_idx == 0)  ───────────────────────
        idx0 = pred_idx_token == 0
        if idx0.any():
            model_out.update(self.log_topk_during_train(logits[idx0], labels[idx0]))

        return model_out

    @torch.no_grad()
    def predict(self, item_seq, time_seq, all_item_feature, all_item_tags, target_tags, save_for_eval=False):
        # all_item_feature.shape: [all_candidate_items, d_model]
        wandb_logs = dict()

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_id_proj_tower(self.item_embedding(item_seq))
        item_emb = item_emb + position_embedding

        if self.skip_hstu:
            output_embs = item_emb
        else:
            attention_mask = self.get_attention_mask(item_seq)
            output_embs = self._hstu(
                x=item_emb,
                attention_mask=attention_mask
            )

        # Calculate multi-interest user embeddings using self-attention
        # attention_net input shape: (batch_size, seq_len, embedding_dim)
        # attention_net output shape: (batch_size, seq_len, num_interest)
        attention_weights = self.attention_net(output_embs)
        
        # Transpose for matmul: (batch_size, num_interest, seq_len)
        attention_weights = attention_weights.permute(0, 2, 1)
        
        # Softmax over the sequence length dimension
        # attention_weights = F.softmax(attention_weights, dim=-1)
        # CRITICAL FIX: Apply masking during inference.
        # Create mask (B, L). Assumes padding index is 0.
        mask = (item_seq != 0)
        # Expand mask (B, K, L)
        mask_expanded = mask.unsqueeze(1).expand(-1, self.num_interest, -1)

        # Apply stable masked softmax
        attention_weights = torch.where(mask_expanded, attention_weights, torch.finfo(attention_weights.dtype).min)
        attention_weights = F.softmax(attention_weights, dim=-1)
        # Handle potential NaNs if sequence is entirely masked
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Calculate user interest embeddings
        # (batch, num_interest, seq_len) @ (batch, seq_len, dim) -> (batch, num_interest, dim)
        full_seq_output = torch.matmul(attention_weights, output_embs)

        if save_for_eval:
            next_token_embedding = output_embs[:, -1]
            saved_user_embs = next_token_embedding.float().cpu().numpy()
            del next_token_embedding
            saved_head_embs = full_seq_output.float().cpu().numpy()
        else:
            saved_user_embs = None
            saved_head_embs = None
       
        del output_embs
        # full_seq_output shape: (b_sz, num_interest, d_model)
        full_seq_output = full_seq_output.float()
        final_seq_output = F.normalize(full_seq_output, p=2, dim=-1)  # Normalize

        # Normalize item_feature for cosine similarity
        all_item_feature = all_item_feature.float()
        all_item_feature = F.normalize(all_item_feature, p=2, dim=-1)
        # all_item_feature.shape: [all_candidate_items, d_model]

        # Perform similarity calculation with each generated token
        similarity_scores = torch.matmul(final_seq_output, all_item_feature.t())
        print('similarity_scores.shape: ', similarity_scores.shape)
        # similarity_scores shape: (b_sz, num_interest, all_items)
        return similarity_scores, wandb_logs, saved_user_embs, saved_head_embs

    @torch.no_grad()
    def compute_item_all(self):
        weight = self.item_id_proj_tower(self.item_embedding.weight)
        return F.normalize(weight, p=2, dim=-1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        # extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask
