from REC.model.basemodel import BaseModel, all_gather
from REC.utils.enum_type import InputType
# Import truncated_normal for initialization consistency
try:
    from REC.model.IDNet.hstu import truncated_normal
except ImportError:
    def truncated_normal(data, mean=0.0, std=0.02):
        torch.nn.init.trunc_normal_(data, mean=mean, std=std)

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import math

EPS = 1e-10

ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "silu": nn.SiLU(),
    "relu": nn.ReLU(),
}

class DualVAE(BaseModel):
    """
    Sequential adaptation of DualVAE (SeqDualVAE) with Causal Attention Pooling, KL Annealing, 
    Information Dropout, and Orthogonality Constraints.
    """
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super().__init__()
        self.logger = getLogger()
        self.item_num = dataload.item_num
        self.max_seq_length: int = config['MAX_ITEM_LIST_LENGTH']
        self.pred_len = config['pred_len']

        if self.pred_len != 1:
            raise NotImplementedError("Causal training currently supports pred_len=1.")

        # --- DualVAE Hyperparameters (Prefix: vae_) ---
        self.k = config.get('vae_latent_dim', 32)
        self.a = config.get('vae_num_aspects', 5)
        
        # --- KL Annealing Setup ---
        self.target_beta_kl = config.get('vae_beta_kl', 0.1) 
        self.kl_anneal_steps = config.get('vae_kl_anneal_steps', 10000) 
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))
        # --------------------------

        self.gama_cl = config.get('vae_gama_cl', 0.01)
        self.cl_temp = 0.2
        self.T_aspect = config.get('vae_aspect_temperature', 0.5) 
        
        # NEW: Orthogonality Constraint Weight
        self.ortho_lambda = config.get('vae_ortho_lambda', 0.1)
        
        vae_encoder_structure_size = config.get('vae_encoder_structure_size', 'small')
        if vae_encoder_structure_size == 'large':
            self.encoder_structure = [256, 128, 64]
        elif vae_encoder_structure_size == 'medium':
            self.encoder_structure = [128, 64]
        else:
            self.encoder_structure = [64]
        self.act_fn_name = config.get('vae_act_fn', 'tanh')
        self.act_fn = ACT.get(self.act_fn_name, nn.Tanh())

        # --- Framework Hyperparameters ---
        self.embedding_dim = config['item_embedding_size'] # D
        self.dropout_rate = config.get('hidden_dropout_prob', 0.2)
        self.loss_type = config.get('loss', 'nce')
        
        # NEW: Information (Latent) Dropout rate
        self.latent_dropout_rate = config.get('vae_latent_dropout', 0.2) 


        # --- Component Initialization ---
        
        # 1. Embeddings and Positional Encoding
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_dim)
        self.input_layernorm = nn.LayerNorm(self.embedding_dim, eps=1e-12)
        self.input_dropout = nn.Dropout(self.dropout_rate)

        # 2. DualVAE Components
        self.item_proj = nn.Linear(self.embedding_dim, self.k * self.a)
        self.item_topics = nn.Parameter(torch.empty(self.a, self.k))

        # Attention Pooling Mechanism
        attn_hidden_dim = max(16, self.k // 2)
        self.attention_pooling_net = nn.Sequential(
            nn.Linear(self.k, attn_hidden_dim),
            self.act_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(attn_hidden_dim, 1, bias=False)
        )

        # VAE Inference Network (DVI Module)
        self.inference_net = self._build_vae_network([self.k] + self.encoder_structure)
        self.user_mu = nn.Linear(self.encoder_structure[-1], self.k)
        self.user_std = nn.Linear(self.encoder_structure[-1], self.k)

        # NEW: Latent Dropout Layer
        self.latent_dropout = nn.Dropout(self.latent_dropout_rate)

        # 3. Loss components (NCE setup)
        if self.loss_type == 'nce':
            if config.get('fix_temp', False):
                 self.register_buffer("logit_scale", torch.tensor(np.log(1 / 0.05)))
            else:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.05))
        else:
            raise NotImplementedError(f"loss={self.loss_type} is not supported.")

        self._verbose = True
        self.reset_params()

    # MODIFICATION: Added Layer Normalization
    def _build_vae_network(self, structure):
        network = nn.Sequential()
        for i in range(len(structure) - 1):
            output_dim = structure[i+1]
            network.add_module(f"fc{i}", nn.Linear(structure[i], output_dim))
            # Add LayerNorm before activation for stability and regularization
            network.add_module(f"ln{i}", nn.LayerNorm(output_dim, eps=1e-12))
            network.add_module(f"act{i}", self.act_fn)
            network.add_module(f"dropout{i}", nn.Dropout(self.dropout_rate))
        return network

    def reset_params(self):
        # (Remains the same - Xavier for VAE, Truncated Normal for Embeddings)
        nn.init.kaiming_uniform_(self.item_topics, a=np.sqrt(5))

        for name, params in self.named_parameters():
            if 'item_topics' in name or 'logit_scale' in name:
                continue

            is_vae_component = ('inference_net' in name or 'user_mu' in name or 'user_std' in name or 'attention_pooling_net' in name)
            
            if is_vae_component:
                if params.dim() > 1:
                    nn.init.xavier_uniform_(params)
                elif 'bias' in name:
                    nn.init.zeros_(params)
                continue

            try:
                truncated_normal(params.data, mean=0.0, std=0.02)
            except Exception as e:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params. Error: {e}")

    def debug_str(self) -> str:
        return f"SeqDualVAE_CausalAttnPool-K{self.k}-A{self.a}-BKL{self.target_beta_kl}-GCL{self.gama_cl}-T{self.T_aspect}-Ortho{self.ortho_lambda}-LDrop{self.latent_dropout_rate}"

    def reparameterize(self, mu, std):
        eps = torch.randn_like(mu)
        return mu + eps * std

    def _process_sequence(self, seq_items):
        # (Remains the same)
        L = seq_items.size(1)
        seq_embs = self.item_embedding(seq_items)
        
        L_pos = min(L, self.max_seq_length)
        position_ids = torch.arange(L_pos, device=seq_items.device).unsqueeze(0).expand(seq_items.shape[0], L_pos)
        pos_embs = self.position_embedding(position_ids)

        if L > L_pos:
             pos_embs = F.pad(pos_embs, (0, 0, 0, L - L_pos))
        elif L < self.max_seq_length:
             if pos_embs.size(1) != L:
                 pos_embs = pos_embs[:, :L, :]
        
        input_seq_embs = seq_embs + pos_embs
        input_seq_embs = self.input_layernorm(input_seq_embs)
        input_seq_embs = self.input_dropout(input_seq_embs)
        return input_seq_embs

    def get_disentangled_item_embs(self, item_ids=None, sequence_embs=None):
        # (Remains the same)
        if sequence_embs is not None:
            embs = sequence_embs
        elif item_ids is not None:
            embs = self.item_embedding(item_ids)
        else:
            raise ValueError("Either item_ids or sequence_embs must be provided.")
            
        projected = self.item_proj(embs)
        return projected.view(*projected.shape[:-1], self.a, self.k)

    def calculate_aspect_probabilities(self, disentangled_embs):
        # (Remains the same)
        norm_embs = F.normalize(disentangled_embs, p=2, dim=-1)
        norm_topics = F.normalize(self.item_topics, p=2, dim=-1)

        if disentangled_embs.dim() == 4: # (B, L, A, K)
             aspect_sim = torch.einsum('blak,ak->bla', norm_embs, norm_topics)
        elif disentangled_embs.dim() == 3: # (B, A, K) or (N, A, K)
             aspect_sim = torch.einsum('nak,ak->na', norm_embs, norm_topics)
        else:
             aspect_sim = torch.sum(torch.mul(norm_embs, norm_topics), dim=-1)
        
        return torch.softmax(aspect_sim / self.T_aspect, dim=-1)

    def contrast_loss(self, z_u_norm, o_u_norm):
        # (Remains the same)
        x = z_u_norm
        x_ = o_u_norm

        pos_score = torch.sum(torch.mul(x_, x), dim=-1)
        pos_score = torch.exp(pos_score / self.cl_temp)

        acl_score = torch.bmm(x_, x.transpose(1, 2))

        diag_mask = torch.eye(self.a, device=x.device).unsqueeze(0).bool()
        acl_score = acl_score.masked_fill(diag_mask, torch.finfo(acl_score.dtype).min)
        
        acl_score = torch.sum(torch.exp(acl_score / self.cl_temp), dim=-1)

        neg_score = acl_score
        info_nec_loss = torch.log(pos_score / (neg_score + EPS))
        info_nec_loss = -torch.mean(info_nec_loss)
        return info_nec_loss

    # NEW: Orthogonality Loss
    def orthogonality_loss(self):
        """Calculates the orthogonality constraint on item_topics."""
        # Normalize the prototypes (A, K)
        norm_topics = F.normalize(self.item_topics, p=2, dim=-1)
        # Calculate similarity matrix (A, A)
        similarity_matrix = torch.matmul(norm_topics, norm_topics.t())
        # Create Identity matrix
        identity = torch.eye(self.a, device=similarity_matrix.device)
        # Calculate the difference (Frobenius norm). We want the off-diagonal elements to be zero.
        loss = torch.norm(similarity_matrix - identity, p='fro')
        return loss

    def _infer_causal_user_representation(self, input_seq_embs, seq_mask):
        """
        Helper to infer z_u causally using windowing (unfold), Attention Pooling, and VAE.
        """
        B, L, D = input_seq_embs.shape
        A, K = self.a, self.k

        # 1. Disentanglement and Cross-Filtering
        seq_dis_embs = self.get_disentangled_item_embs(sequence_embs=input_seq_embs)
        seq_aspect_probs = self.calculate_aspect_probabilities(seq_dis_embs)
        filtered_embs = seq_dis_embs * seq_aspect_probs.unsqueeze(-1)

        # 2. Causal Aggregation (Attention Pooling) using Unfold
        # (Setup for unfolding remains the same)
        filtered_embs_reshaped = filtered_embs.permute(0, 2, 1, 3).reshape(B*A, L, K)
        padded_embs = F.pad(filtered_embs_reshaped, (0, 0, L - 1, 0), "constant", 0)
        causal_windows = padded_embs.unfold(dimension=1, size=L, step=1).permute(0, 1, 3, 2)

        seq_mask_expanded = seq_mask.unsqueeze(1).repeat(1, A, 1).reshape(B*A, L)
        padded_mask = F.pad(seq_mask_expanded, (L - 1, 0), "constant", False)
        causal_mask_windows = padded_mask.unfold(dimension=1, size=L, step=1)

        attn_input = causal_windows.reshape(B*A*L, L, K)
        mask_flat = causal_mask_windows.reshape(B*A*L, L)

        # Calculate attention scores and apply Masked Softmax
        attn_scores = self.attention_pooling_net(attn_input).squeeze(-1)
        attn_weights = attn_scores.masked_fill(~mask_flat, torch.finfo(attn_scores.dtype).min)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Aggregate
        aggregated_input_flat = torch.sum(attn_weights.unsqueeze(-1) * attn_input, dim=1)
        
        # 3. VAE Inference
        h = self.inference_net(aggregated_input_flat)
        mu = self.user_mu(h)
        # Use Softplus for stability
        std = F.softplus(self.user_std(h)) + 1e-4

        # KL Divergence Calculation
        kl_a = -0.5 * (1 + 2.0 * torch.log(std + EPS) - mu.pow(2) - std.pow(2))
        kl_reshaped = kl_a.sum(dim=-1).view(B, A, L).permute(0, 2, 1)

        # Reparameterization
        if self.training:
            z_u_a = self.reparameterize(mu, std)
            # NEW: Apply Information Dropout (Variational Dropout)
            z_u_a = self.latent_dropout(z_u_a)
        else:
            # Use deterministic mean during evaluation
            z_u_a = mu
        
        # Reshape z_u back to (B, L, A, K)
        z_u_causal = z_u_a.view(B, A, L, K).permute(0, 2, 1, 3)
        
        return z_u_causal, kl_reshaped


    def forward(self, interaction):
        
        # === KL Annealing Calculation ===
        if self.training:
            self.global_step.add_(1)
            if self.kl_anneal_steps > 0:
                # Linear annealing schedule
                anneal_factor = min(1.0, self.global_step.item() / self.kl_anneal_steps)
                current_beta_kl_value = self.target_beta_kl * anneal_factor
            else:
                current_beta_kl_value = self.target_beta_kl
        else:
            current_beta_kl_value = self.target_beta_kl
        # ================================

        # Input unpacking
        items, neg_items, user_attention_mask, _ = interaction
        user_attention_mask = user_attention_mask.bool()

        L = self.max_seq_length
        
        # Define Inputs (x_0...x_{L-1}) and Targets (x_1...x_L) for Causal Training
        seq_items = items[:, :L]
        pos_target_items = items[:, 1:L+1] 
        
        seq_mask = user_attention_mask[:, :L]
        # Mask for valid targets (B, L)
        target_mask = user_attention_mask[:, 1:L+1]

        # Process Sequence (Positional Embeddings + Regularization)
        input_seq_embs = self._process_sequence(seq_items)

        # Infer Causal User Representation
        # z_u_causal: (B, L, A, K), kl_reshaped: (B, L, A)
        z_u_causal, kl_reshaped = self._infer_causal_user_representation(input_seq_embs, seq_mask)

        # Calculate KL loss
        mask_dtype = target_mask.unsqueeze(-1).to(kl_reshaped.dtype) # (B, L, 1)
        kl_loss = (kl_reshaped * mask_dtype).sum() / ((mask_dtype.sum() * self.a) + EPS)


        # 3. Prepare Target Embeddings (Positive)
        pos_target_dis_embs = self.get_disentangled_item_embs(item_ids=pos_target_items)
        pos_target_aspect_probs = self.calculate_aspect_probabilities(pos_target_dis_embs)

        # 4. Handle Negatives
        neg_ids = neg_items[:, -1]
        
        neg_embs = self.item_embedding(neg_ids)
        neg_proj = self.item_proj(neg_embs)

        D_proj = neg_proj.shape[-1]
        neg_proj_all = all_gather(neg_proj.reshape(-1, D_proj), sync_grads=True)
        neg_dis_embs_all = neg_proj_all.view(-1, self.a, self.k)
        neg_target_aspect_probs = self.calculate_aspect_probabilities(neg_dis_embs_all)


        # 5. JG Module Adaptation: Calculate Scores (Normalized)
        
        # Normalize representations
        z_u_norm = F.normalize(z_u_causal, p=2, dim=-1)
        pos_target_dis_embs_norm = F.normalize(pos_target_dis_embs, p=2, dim=-1)
        neg_dis_embs_all_norm = F.normalize(neg_dis_embs_all, p=2, dim=-1)

        # Calculate Positive interactions
        pos_interactions = torch.sum(z_u_norm * pos_target_dis_embs_norm, dim=-1)
        weighted_pos_interactions = pos_interactions * pos_target_aspect_probs
        pos_logits = torch.sum(weighted_pos_interactions, dim=-1).unsqueeze(-1)

        # Calculate Negative Logits
        neg_interactions = torch.einsum('blak,nak->blna', z_u_norm, neg_dis_embs_all_norm)
        weighted_neg_interactions = neg_interactions * neg_target_aspect_probs.unsqueeze(0).unsqueeze(0)
        neg_logits = torch.sum(weighted_neg_interactions, dim=-1)

        # 6. Loss Calculation (Causal NCE + KL + CL + Ortho)
        model_out = defaultdict(float)

        # NCE Loss
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))
        logit_scale = self.logit_scale.exp()

        logits = torch.cat([pos_logits, neg_logits], dim=-1) * logit_scale
        
        # Flatten and Mask
        valid_logits = logits[target_mask]

        if valid_logits.shape[0] > 0:
            labels = torch.zeros(valid_logits.shape[0], dtype=torch.int64, device=z_u_causal.device)
            nce_loss = F.cross_entropy(valid_logits, labels)
        else:
            nce_loss = torch.tensor(0.0, device=z_u_causal.device, dtype=z_u_causal.dtype)

        # NRC Contrastive Loss (CL)
        z_u_norm_valid = z_u_norm[target_mask]
        pos_target_norm_valid = pos_target_dis_embs_norm[target_mask]

        if z_u_norm_valid.shape[0] > 0:
            cl_loss = self.contrast_loss(z_u_norm_valid, pos_target_norm_valid)
        else:
            cl_loss = torch.tensor(0.0, device=z_u_causal.device, dtype=z_u_causal.dtype)

        # NEW: Orthogonality Loss
        ortho_loss = self.orthogonality_loss()

        # Total Loss (Using the annealed beta_kl)
        current_beta_kl = torch.tensor(current_beta_kl_value, dtype=kl_loss.dtype, device=kl_loss.device)
        
        total_loss = nce_loss + current_beta_kl * kl_loss + self.gama_cl * cl_loss + self.ortho_lambda * ortho_loss

        model_out["loss"] = total_loss

        # Add top-k metrics and monitoring
        if valid_logits.shape[0] > 0:
            model_out.update(self.log_topk_during_train(valid_logits, labels))
        
        model_out['kl_loss'] = current_beta_kl * kl_loss.detach()
        model_out['ortho_loss'] = self.ortho_lambda * ortho_loss.detach()
        model_out['current_beta_kl'] = current_beta_kl.detach()
        model_out['cl_loss'] = self.gama_cl * cl_loss.detach()

        return model_out

    # (log_topk_during_train, predict, compute_item_all remain the same as the previous Causal version)
    # ... (Include the remaining methods here - omitted for brevity) ...

    def log_topk_during_train(self, logits, labels):
        # (Remains the same)
        log_dict = {}
        log_dict['nce_samples'] = (logits > torch.finfo(logits.dtype).min / 100).sum(
            dim=1).float().mean().detach()
        
        for k in [1, 5, 10, 50, 100]:
            if k > logits.size(-1):
                break
            indices = logits.topk(k, dim=-1).indices
            log_dict[f'nce_top{k}_acc'] = labels.view(-1, 1).eq(indices).any(dim=-1).float().mean().detach()
        return log_dict


    @torch.no_grad()
    def predict(self, item_seq, time_seq=None, all_item_feature=None, all_item_tags=None, target_tags=None, save_for_eval=False):
        self.eval()
        wandb_logs = dict()

        # 1. Prepare inputs
        seq_mask = (item_seq != 0).bool()

        # Process Sequence
        input_seq_embs = self._process_sequence(item_seq)

        # Infer Causal User Representation
        # z_u_causal: (B, L, A, K)
        # self.training is False due to self.eval(), so VAE uses mean (mu) and latent_dropout is inactive.
        z_u_causal, _ = self._infer_causal_user_representation(input_seq_embs, seq_mask)

        # Select the representation corresponding to the last item for prediction
        seq_lens = seq_mask.sum(dim=1)
        last_idx = (seq_lens - 1).clamp(min=0) # (B,)
        
        # Gather the last representation: (B, A, K)
        B = item_seq.shape[0]
        batch_indices = torch.arange(B, device=item_seq.device)
        # Select z_u[b, last_idx[b], :, :]
        z_u = z_u_causal[batch_indices, last_idx]


        # Normalize User Representation for MIPS
        z_u_norm = F.normalize(z_u, p=2, dim=-1)
        z_u_combined = z_u_norm.view(z_u.shape[0], self.a * self.k)

        # 4. Retrieve Combined Item Embeddings
        if all_item_feature is not None:
             if all_item_feature.shape[1] != self.a * self.k:
                 self.logger.warning("all_item_feature dimension mismatch. Recomputing.")
                 combined_item_embs = self.compute_item_all()
             else:
                combined_item_embs = all_item_feature
        else:
            combined_item_embs = self.compute_item_all()

        # 5. Calculate Scores (Simple MIPS)
        similarity_scores = torch.matmul(z_u_combined, combined_item_embs.t())
        
        # Apply logit_scale (Consistency Fix)
        if self.loss_type == 'nce':
            self.logit_scale.clamp_(0, np.log(100))
            logit_scale = self.logit_scale.exp()
            similarity_scores = similarity_scores * logit_scale
        
        # Reshape to (B, 1, N_items).
        similarity_scores = similarity_scores.unsqueeze(1)

        # 6. Prepare return values
        saved_user_embs = None
        saved_head_embs = None

        if save_for_eval:
            # Use .float() for compatibility when converting to CPU/Numpy
            saved_user_embs = z_u_combined.float().cpu().numpy()
            saved_head_embs = z_u_norm.float().cpu().numpy()

        return similarity_scores, wandb_logs, saved_user_embs, saved_head_embs

    @torch.no_grad()
    def compute_item_all(self):
        # (Remains the same)
        # 1. Get base disentangled embeddings
        all_item_ids = torch.arange(self.item_num, device=self.item_embedding.weight.device)
        all_dis_embs = self.get_disentangled_item_embs(item_ids=all_item_ids)

        # 2. Get aspect probabilities
        all_aspect_probs = self.calculate_aspect_probabilities(all_dis_embs)

        # Normalize embeddings before weighting
        all_dis_embs_norm = F.normalize(all_dis_embs, p=2, dim=-1)

        # 3. Integrate (Apply weighting)
        weighted_embs = all_dis_embs_norm * all_aspect_probs.unsqueeze(-1)

        # 4. Concatenate (Reshape)
        combined_embs = weighted_embs.view(self.item_num, self.a * self.k)

        # No final normalization
        return combined_embs
