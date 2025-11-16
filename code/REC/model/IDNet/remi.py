from REC.model.basemodel import BaseModel, all_gather
from REC.model.IDNet.hstu import HSTUJagged, SequentialTransductionUnitJagged, RelativeBucketedTimeAndPositionBasedBias, truncated_normal
from REC.utils.enum_type import InputType

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class REMI(BaseModel):
    """
    Implements REMI (Rethinking Multi-Interest Learning) based on ComiRec-SA architecture and HSTU.
    REMI enhances the training scheme with Interest-aware Hard Negative mining (IHN)
    and Routing Regularization (RR).
    Paper: https://arxiv.org/abs/2302.14532
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
        self.input_dropout_cfg = config.get('input_dropout', False)

        # --- REMI Specific Parameters ---
        # Lambda for Routing Regularization (RR). Paper suggests tuning around 1e2.
        self.lambda_rr = config.get('lambda_rr', 100.0)
        # Beta for Interest-aware Hard Negative mining (IHN). If <= 0, standard NCE is used.
        self.beta_ihn = config.get('beta_ihn', 1.0)
        self.logger.info(f"REMI initialized with lambda_rr={self.lambda_rr}, beta_ihn={self.beta_ihn}")

        # --- HSTU Configuration (Same as ComiRec) ---
        self._num_blocks: int = config['n_layers']
        self._num_heads: int = config['n_heads']
        self._dqk: int = config['hstu_embedding_size'] // config['n_heads']
        self._dv: int = config['hstu_embedding_size'] // config['n_heads']
        self._linear_activation: str = config['hidden_act'] if config['hidden_act'] else "silu"
        self._linear_dropout_rate: float = config['hidden_dropout_prob']
        self._attn_dropout_rate: float = config['attn_dropout_prob']
        self._enable_relative_attention_bias: bool = config.get('enable_relative_attention_bias', False)
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
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=self.max_seq_length * 2,
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

        self.input_dropout = nn.Dropout(self._linear_dropout_rate) if self.input_dropout_cfg else nn.Identity()

        self.input_dropout = nn.Dropout(self._linear_dropout_rate) if self.input_dropout else nn.Identity()
        self.interest_hidden = config.get('interest_hidden', int(self._hstu_embedding_dim * config.get('interest_hidden_ratio', 0.5)))
        self.num_interest = config.get('interest_num', 4)
        # Self-attention network to generate interest capsules
        self.attention_net = nn.Sequential(
            nn.Linear(self._hstu_embedding_dim, self.interest_hidden, bias=config.get('attention_net_bias', True)),
            nn.Tanh(),
            nn.Dropout(self._linear_dropout_rate), 
            nn.Linear(self.interest_hidden, self.num_interest, bias=False)
        )

        # --- Embeddings and Loss ---
        self.item_embedding = nn.Embedding(self.item_num, self._item_embedding_dim, padding_idx=0)
        self.item_id_proj_tower = nn.Identity() if self._item_embedding_dim == self._hstu_embedding_dim else nn.Linear(self._item_embedding_dim, self._hstu_embedding_dim, bias=False)
        self.loss = config['loss']
        if self.loss in ['nce']:
            if config.get('fix_temp', False):
                self.logger.info(f"Fixed logit_scale 20")
                self.register_buffer("logit_scale", torch.tensor(np.log(1 / 0.05)))
            else:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.05))
            self.nce_thres = config.get('nce_thres', 0.99)
            self.logger.info(f"nce threshold setting to {self.nce_thres}")
        else:
            raise NotImplementedError(f"loss={self.loss} is not supported")

        horizon_discount = torch.tensor([self.medusa_lambda ** pred_idx for pred_idx in range(self.pred_len)])
        sum_discount = sum(horizon_discount)
        self.register_buffer('horizon_discount', horizon_discount / sum_discount)
        self.eval_pred_len = config.get('eval_pred_len')
        self.rank = torch.distributed.get_rank()

        # causal forward mask
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (self.max_seq_length, self.max_seq_length),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = True
        self.reset_params()

    def reset_params(self):
        # Initialization (similar to ComiRec)
        for name, params in self.named_parameters():
            if ("_hstu" in name) or ("_embedding_module" in name) or ('logit_scale' in name):
                continue
            try:
                truncated_normal(params.data, mean=0.0, std=0.02)
            except:
                pass

    def debug_str(self) -> str:
        debug_str = (
            f"REMI-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if self.beta_ihn > 0:
             debug_str += f"-IHN{self.beta_ihn}"
        if self.lambda_rr > 0:
            debug_str += f"-RR{self.lambda_rr}"
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def calculate_rr_loss(self, attention, mask):
        """
        REMI Component 1: Routing Regularization (RR) Loss.
        Calculates the RR loss using masked statistics to handle padded sequences correctly.
        L_reg = || diag(C) ||_F^2, where C is the covariance matrix of the routing weights.
        
        Args:
            attention (torch.Tensor): Routing matrix A. Shape: (Batch, K, L_seq).
            mask (torch.Tensor): Boolean mask for valid items. Shape: (Batch, K, L_seq).
        
        Returns:
            torch.Tensor: RR loss per batch item. Shape: (Batch,).
        """
        # attention: (Batch, K, L_seq), mask: (Batch, K, L_seq).
        mask_float = mask.float()
        
        # 1. Calculate Masked Mean
        # Count the number of valid items per sequence/interest
        seq_lens = mask_float.sum(dim=-1, keepdim=True).clamp(min=1.0) # (Batch, K, 1)
        
        # Ensure attention is zeroed out where masked (Softmax should handle this, but for safety)
        attention_masked = attention * mask_float
        
        attention_sum = attention_masked.sum(dim=-1, keepdim=True)
        C_mean = attention_sum / seq_lens # (Batch, K, 1)

        # 2. Calculate Deviation from Mean (centered weights)
        # We must apply the mask again so that (A - Mean) is 0 for padded items.
        C_dev = (attention_masked - C_mean) * mask_float # (Batch, K, L_seq)

        # 3. Calculate the covariance matrix C = C_dev @ C_dev^T. (Batch, K, K)
        # Normalization by hidden_size follows the reference REMI implementation.
        C_cov = torch.bmm(C_dev, C_dev.transpose(1, 2)) / self._hstu_embedding_dim

        # 4. Extract the diagonal elements (variances). (Batch, K)
        variances = torch.diagonal(C_cov, dim1=-2, dim2=-1)

        # 5. Calculate the square of the Frobenius norm of diag(C).
        # L2 norm squared of the variances vector.
        norm_sq = torch.norm(variances, dim=1) ** 2 # (Batch,)

        return norm_sq

    def ihn_loss(self, cur_embs_valid, target_pos_valid, neg_embedding_all):
        """
        REMI Component 2: Interest-aware Hard Negative Mining (IHN) Loss.
        Uses importance sampling to approximate an ideal hard negative distribution.
        Implemented in log-space for numerical stability and adapted to the existing NCE framework (normalization/scaling).
        """
        beta = self.beta_ihn

        # Use normalized embeddings for cosine similarity as scores
        output_embs = F.normalize(cur_embs_valid, p=2, dim=-1)
        target_pos_embs = F.normalize(target_pos_valid, p=2, dim=-1)
        neg_embedding_all_norm = F.normalize(neg_embedding_all, p=2, dim=-1)
        neg_embedding_all_t = neg_embedding_all_norm.T.contiguous()

        # Calculate scores
        pos_scores = (output_embs * target_pos_embs).sum(-1, keepdim=True)
        neg_scores = output_embs @ neg_embedding_all_t

        # Handle temperature/logit_scale
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))
        logit_scale = self.logit_scale.exp()

        # Scaled scores (logits)
        pos_logits = pos_scores * logit_scale
        neg_logits = neg_scores * logit_scale

        # Masking for accidental hits
        fix_logits = target_pos_embs @ neg_embedding_all_t
        mask = fix_logits > self.nce_thres
        # Set masked logits to a very small number before exp().
        masked_neg_logits = neg_logits.masked_fill(mask, torch.finfo(neg_logits.dtype).min)

        # Prepare standard logits and labels for logging metrics
        standard_logits = torch.cat([pos_logits, masked_neg_logits], dim=-1)
        labels = torch.zeros(standard_logits.size(0), device=standard_logits.device, dtype=torch.int64)

        if beta <= 0:
            # Uniform sampling (Standard NCE). Use F.cross_entropy for efficiency.
            tok_loss = F.cross_entropy(standard_logits, labels, reduction='none')
            return tok_loss, standard_logits, labels

        else:
            # IHN implementation (Stable Log-Space version)
            # We want to calculate Loss = log(exp(s_pos) + Neg_IHN) - s_pos
            # Where Neg_IHN = sum(exp((beta+1)s_neg)) / mean(exp(beta*s_neg))

            N_neg = masked_neg_logits.size(1)
            if N_neg == 0:
                 # Handle case with no negative samples
                tok_loss = torch.zeros_like(pos_logits.squeeze(-1))
                return tok_loss, standard_logits, labels

            log_N_neg = torch.log(torch.tensor(N_neg, device=masked_neg_logits.device, dtype=masked_neg_logits.dtype))

            # 1. Calculate log(Numerator) = logsumexp((beta+1) * s_neg)
            logits_beta_plus_1 = (beta + 1) * masked_neg_logits
            log_numerator = torch.logsumexp(logits_beta_plus_1, dim=1, keepdim=True)

            # 2. Calculate log(Z_beta) = log(mean(exp(beta*s_neg))) = logsumexp(beta*s_neg) - log(N_neg)
            logits_beta = beta * masked_neg_logits
            log_sum_imp = torch.logsumexp(logits_beta, dim=1, keepdim=True)
            log_Z_beta = log_sum_imp - log_N_neg

            # 3. Calculate log(Neg_IHN) = log_numerator - log_Z_beta
            # Handle potential division by zero if Z_beta is 0 (log_Z_beta = -inf)
            valid_imp = log_Z_beta > torch.finfo(log_Z_beta.dtype).min
            log_Neg_IHN = torch.full_like(log_numerator, float('-inf')) # Initialize to log(0)
            if valid_imp.any():
                log_Neg_IHN[valid_imp] = log_numerator[valid_imp] - log_Z_beta[valid_imp]

            # 4. Calculate log_denominator = log( exp(s_pos) + exp(log_Neg_IHN) )
            # Use torch.logaddexp for stability: logaddexp(x, y) = log(exp(x) + exp(y))
            log_denominator = torch.logaddexp(pos_logits, log_Neg_IHN)

            # 5. Final loss = log_denominator - s_pos
            tok_loss = log_denominator.squeeze(-1) - pos_logits.squeeze(-1)

            return tok_loss, standard_logits, labels


    def log_topk_during_train(self, logits, labels):
        # Same as ComiRec
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

        # ─── (1)  SEQUENCE ENCODING ──────────────────────────────────────────────────
        pos_items_embs = self.item_id_proj_tower(self.item_embedding(items))  # [batch, seq_len+pred_len, dim]
        input_emb = pos_items_embs[:, :-self.pred_len, :]  # [batch, seq_len, dim]

        # Positional Embeddings
        position_ids = torch.arange(input_emb.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(user_attention_mask[:, :-self.pred_len])
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_emb + position_embedding

        # HSTU Blocks
        if self.skip_hstu:
            output_embs = self.input_dropout(input_emb)
        else:
            attention_mask = self.get_attention_mask(user_attention_mask[:, :-self.pred_len])
            output_embs = self._hstu(x=input_emb, attention_mask=attention_mask)

        model_out = defaultdict(float)
        B = output_embs.shape[0]
        dtype = output_embs.dtype
        L = self.max_seq_length
        P = self.pred_len
        D = output_embs.size(-1)
        K = self.num_interest

        # Prepare negative embeddings (All Gather)
        neg_embedding = self.item_id_proj_tower(self.item_embedding(neg_items[:, -1]))
        # Normalization happens inside ihn_loss
        neg_embedding_all = all_gather(neg_embedding, sync_grads=True).reshape(-1, D)  # [total_num_negs, dim]

        # ─── (2) CAUSAL MULTI-INTEREST EXTRACTION ───────────────────────────────────
        # Setup for causal processing (unfolding windows)
        padded_output_embs = F.pad(output_embs, (0, 0, L - 1, 0), "constant", 0)
        context_windows = padded_output_embs.unfold(dimension=1, size=L, step=1).permute(0, 1, 3, 2)
        # context_windows: (B, L, L, D)

        context_mask = user_attention_mask[:, :-self.pred_len] # (B, L) - Mask for valid time steps
        padded_mask = F.pad(context_mask, (L - 1, 0), "constant", 0)
        context_windows_mask = padded_mask.unfold(dimension=1, size=L, step=1) # (B, L, L) - Mask for items within windows

        # Flatten batch and time dimensions for parallel processing: (B*L, L, D)
        attn_input = context_windows.reshape(B*L, L, D)
        
        # Apply attention net (Routing): (B*L, L, D) -> (B*L, L, K)
        attn_weights_raw = self.attention_net(attn_input)
        # Transpose: (B*L, K, L)
        attn_weights = attn_weights_raw.permute(0, 2, 1)

        # Masking and Softmax
        # Reshape and expand mask: (B*L, L) -> (B*L, K, L)
        mask_reshaped = context_windows_mask.reshape(B*L, L)
        mask_expanded = mask_reshaped.unsqueeze(1).expand(-1, K, -1)

        # Apply mask using a large negative number where mask is False.
        attention_weights = torch.where(mask_expanded, attn_weights, torch.finfo(attn_weights.dtype).min)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # --- REMI: Routing Regularization (RR) ---
        if self.lambda_rr > 0:
            # Calculate RR loss on the routing matrix (attention_weights).
            # Pass the mask to ensure correct calculation on padded sequences.
            # rr_loss_per_step shape: (B*L,)
            rr_loss_per_step = self.calculate_rr_loss(attention_weights, mask_expanded)
            
            # Aggregate RR loss: Average only over valid time steps (B, L)
            rr_loss_reshaped = rr_loss_per_step.reshape(B, L)
            # Apply the time step mask (context_mask)
            rr_loss_masked = rr_loss_reshaped * context_mask.float()
            
            # Calculate the average loss
            valid_steps = context_mask.sum().clamp(min=1.0)
            total_rr_loss = rr_loss_masked.sum() / valid_steps

            model_out["rr_loss"] = total_rr_loss.detach()
            model_out["loss"] += self.lambda_rr * total_rr_loss

        # Calculate Causal User Interest Embeddings
        # (B*L, K, L) @ (B*L, L, D) -> (B*L, K, D)
        user_interest_emb_flat = torch.matmul(attention_weights, attn_input)
        # Reshape back to (B, L, K, D)
        user_interest_emb_causal = user_interest_emb_flat.reshape(B, L, K, D)

        # ─── (3)  FUTURE TARGET WINDOWS & MASKS ────────────────────────
        # pos_items_embs: (B, L + P, D)   user_attention_mask: (B, L + P)
        windows_pos = pos_items_embs.unfold(1, P, 1)[:, 1:].permute(0, 3, 1, 2)  # (B, P, L, D)
        windows_mask = user_attention_mask.unfold(1, P, 1)[:, 1:].permute(0, 2, 1)  # (B, P, L)
        base_mask = user_attention_mask[:, :L].unsqueeze(1)  # (B, 1, L)
        final_mask = base_mask & windows_mask  # (B, P, L)

        # ─── (4) INTEREST SELECTION (Hard Readout) ────────────────────────
        # Calculate similarity using the corresponding causal interests.
        # similarity result: (B, P, L, K)
        similarity = torch.einsum('blkd,bpld->bplk', user_interest_emb_causal, windows_pos)

        # "Hard Readout": Select the interest vector with the highest similarity score.
        # We can use argmax directly on similarity as softmax is monotonic.
        best_interest_indices = torch.argmax(similarity, dim=-1) # (B, P, L)

        # Create indices for advanced indexing.
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, P, L)
        l_indices = torch.arange(L, device=device).view(1, 1, L).expand(B, P, -1)

        # Select the final user representation (the selected interest): (B, P, L, D)
        cur_embs = user_interest_emb_causal[batch_indices, l_indices, best_interest_indices]

        # ─── (5)  FLATTEN (batch × pred_idx × seq_len) AXIS  ──────────────────────────────────
        big_batch = B * P
        mask_flat = final_mask.reshape(big_batch, L).bool()
        cur_flat = cur_embs.reshape(big_batch, L, self._hstu_embedding_dim)[mask_flat]    # (N_tok, D)
        pos_flat = windows_pos.reshape(big_batch, L, self._hstu_embedding_dim)[mask_flat] # (N_tok, D)

        # ─── (6)  REMI: Interest-aware Hard Negative Mining (IHN) Loss ──────────────────
        # Calculate token-level loss using IHN strategy (or standard NCE if beta<=0).
        tok_loss, logits, labels = self.ihn_loss(cur_flat, pos_flat, neg_embedding_all)
        # tok_loss: (N_tok,)

        # ─── (7)  LOSS AGGREGATION (Medusa Style) ───────────────────────────────────────────
        # map every token back to its prediction offset 0 … P‑1
        pred_idx_token = (torch.arange(P, device=device).repeat(B)        # (BP ,)
                        .unsqueeze(1).expand(-1, L).reshape(-1)[mask_flat.reshape(-1)])

        # sum & count per offset
        loss_sum_per_p = torch.zeros(self.pred_len, device=device, dtype=torch.float32)
        cnt_per_p = torch.zeros_like(loss_sum_per_p)
        loss_sum_per_p.index_add_(0, pred_idx_token, tok_loss.float())
        cnt_per_p.index_add_(0, pred_idx_token, torch.ones_like(tok_loss, dtype=torch.float32))
        mean_loss_per_p = (loss_sum_per_p / torch.clamp_min(cnt_per_p, 1.0)).to(dtype=dtype)

        # ─── (8)  λ–SCHEDULE & TOTAL LOSS  ────────────────────────────────────────────
        lam_vec = self.horizon_discount.to(dtype=dtype)
        per_pred_loss = lam_vec * mean_loss_per_p  # (P ,)
        # Note: RR loss was already added to model_out["loss"] if applicable.
        model_out["loss"] += per_pred_loss.sum()

        # ─── (9)  OPTIONAL TOP‑k METRICS  (only pred_idx == 0)  ───────────────────────
        idx0 = pred_idx_token == 0
        if idx0.any():
            model_out.update(self.log_topk_during_train(logits[idx0], labels[idx0]))

        return model_out

    @torch.no_grad()
    def predict(self, item_seq, time_seq, all_item_feature, all_item_tags, target_tags, save_for_eval=False):
        # Prediction is identical to ComiRec, as REMI primarily affects the training scheme.
        # We maintain normalization consistency with the training setup.
        wandb_logs = dict()

        # Sequence Encoding
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

        # Multi-Interest Extraction
        attention_weights = self.attention_net(output_embs)

        # Transpose for matmul: (batch_size, num_interest, seq_len)
        attention_weights = attention_weights.permute(0, 2, 1)

        # Masking and Softmax
        mask = (item_seq != 0)
        mask_expanded = mask.unsqueeze(1).expand(-1, self.num_interest, -1)

        # Apply stable masked softmax
        attention_weights = torch.where(mask_expanded, attention_weights, torch.finfo(attention_weights.dtype).min)
        attention_weights = F.softmax(attention_weights, dim=-1)
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
        
        # Scoring
        full_seq_output = full_seq_output.float()
        # Normalize user embeddings (consistency with NCE/IHN training setup)
        final_seq_output = F.normalize(full_seq_output, p=2, dim=-1)

        # Normalize item_feature for cosine similarity
        all_item_feature = all_item_feature.float()
        all_item_feature = F.normalize(all_item_feature, p=2, dim=-1)

        # Perform similarity calculation
        similarity_scores = torch.matmul(final_seq_output, all_item_feature.t())
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
        return extended_attention_mask
