"""
Implements HSTU (Hierarchical Sequential Transduction Unit) in
Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
(https://arxiv.org/abs/2402.17152).
"""

import abc
from collections import defaultdict
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger

from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel, l2_norm, all_gather
from REC.model.llm_heads import ResBlock, Rescale
from REC.model.layers import AsymmetricLoss


def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x


TIMESTAMPS_KEY = "timestamps"


class RelativeAttentionBiasModule(torch.nn.Module):

    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1: N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias


HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor  # [bs, 1, n, n]
):
    B, _, n, _ = attention_mask.size()

    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        q.view(B, n, num_heads, attention_dim),
        k.view(B, n, num_heads, attention_dim),
    )
    qk_attn = F.silu(qk_attn) / n
    qk_attn = qk_attn * attention_mask
    attn_output = torch.einsum(
        "bhnm,bmhd->bnhd",
        qk_attn,
        v.reshape(B, n, num_heads, linear_dim),
    ).reshape(B, n, num_heads * linear_dim)
    return attn_output


class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 2 * num_heads
                        + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads * (3 if concat_ua else 1),
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (\sum_i N_i, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: optional (B, N) x int64.
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
            delta_x_offsets: optional 2-tuple ((B,) x int32, (B,) x int32).
                For the 1st element in the tuple, each element is in [0, x_offsets[-1]). For the
                2nd element in the tuple, each element is in [0, N).
            cache: Optional 4-tuple of (v, padded_q, padded_k, output) from prior runs,
                where all except padded_q, padded_k are jagged.
        Returns:
            x' = f(x), (\sum_i N_i, D) x float.
        """

        normed_x = self._norm_input(x)
        if self._linear_config == "uvqk":
            batched_mm_output = torch.matmul(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        B: int = attention_mask.size(0)
        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            attn_output = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                attention_mask=attention_mask
            )

        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + x
        )

        return new_outputs


class HSTUJagged(torch.nn.Module):

    def __init__(
        self,
        modules: List[SequentialTransductionUnitJagged],
        autocast_dtype: torch.dtype,
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: torch.dtype = autocast_dtype

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
        Returns:
            x' = f(x), (B, N, D) x float
        """

        for i, layer in enumerate(self._attention_layers):
            x = layer(
                x=x,
                attention_mask=attention_mask
            )

        return x


class HSTU(BaseModel):
    """
    Implements HSTU (Hierarchical Sequential Transduction Unit) in
    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,
    https://arxiv.org/abs/2402.17152.

    Note that this implementation is intended for reproducing experiments in
    the traditional sequential recommender setting (Section 4.1.1), and does
    not yet use optimized kernels discussed in the paper.
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

        # when layer=0, head=1, baseline, only predict the next item
        # when layer=0, head>1, autoregressively predict, during train: next one item prediction, during test: number of steps depends on medusa_num_heads
        # when layer=0, steps>1, discounted loss. number of steps depends on pred_len and eval_pred_len
        # when layer=1, if loss=nce, assert medusa_num_heads == pred_len, each head predicts a different step
        # when layer=1, if loss=prior, assert medusa_num_heads == num_prior, each head predicts a different prior/category, and we use discounted loss
        self.medusa_lambda = config['medusa_lambda']
        self.num_segment_head = config['num_segment_head']
        self.num_prior_head = config['num_prior_head']
        self.head_interaction = config['head_interaction']
        if config['head_interaction'] in ['multiplicative', 'hierarchical']:
            self.medusa_num_heads = self.num_segment_head * self.num_prior_head
        elif config['head_interaction'] == 'additive':
            self.medusa_num_heads = self.num_segment_head + self.num_prior_head
        else:
            raise ValueError(f'Unknown head_interaction: {config["head_interaction"]}')
        medusa_num_layers = config['medusa_num_layers']
        self.category_by = config['category_by']

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

        self.item_embedding = nn.Embedding(self.item_num, self._item_embedding_dim, padding_idx=0)
        self.item_id_proj_tower = nn.Identity() if config['item_embedding_size'] == config['hstu_embedding_size'] else nn.Linear(config['item_embedding_size'], config['hstu_embedding_size'], bias=False)
        self.loss = config['loss']
        self.neg_sample_by_cat = config['neg_sample_by_cat']
        if self.loss != 'prior':
            self.neg_sample_by_cat = False
        self.pos_sample_mix_ratio = config['pos_sample_mix_ratio']
        if self.loss in ['nce', 'prior']:
            if config['fix_temp']:
                self.logger.info(f"Fixed logit_scale 20")
                self.register_buffer("logit_scale", torch.tensor(np.log(1 / 0.05)))
            else:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.05))
            self.nce_thres = config['nce_thres'] if config['nce_thres'] else 0.99
            self.logger.info(f"nce threshold setting to {self.nce_thres}")
            self.seg_len = self.pred_len
            if medusa_num_layers > 0:
                assert self.pred_len % self.num_segment_head == 0, \
                    "pred_len must be divisible by the number of segments"
                self.seg_len = self.pred_len // self.num_segment_head
        else:
            raise NotImplementedError(f"loss={self.loss} is not supported")

        horizon_discount = torch.tensor([self.medusa_lambda ** pred_idx for pred_idx in range(self.pred_len)])
        sum_discount = sum(horizon_discount)
        self.register_buffer('horizon_discount', horizon_discount / sum_discount)
        if medusa_num_layers == 0:
            self.medusa_head = nn.ModuleList([nn.Identity() for _ in range(self.medusa_num_heads)])
        else:
            if self.loss in ['nce', 'prior']:
                if self.head_interaction == 'hierarchical':
                    self.head_norm          = config.get("head_norm", False)           # LayerNorm in head MLPs
                    self.cat_bottleneck     = config.get("cat_bottleneck", False)      # 2-layer bottleneck in category head
                    self.cat_bottleneck_dim = config.get("cat_bottleneck_dim", self._hstu_embedding_dim // 2)
                    self.share_seg_weights  = config.get("share_seg_weights", False)  # tie segment heads per category
                    self.use_seg_embed      = config.get("segment_embed", False)       # learned segment embedding

                    if self.use_seg_embed:
                        self.segment_emb = nn.Embedding(self.num_segment_head, self._hstu_embedding_dim)

                    def _make_cat_block():
                        layers = []
                        if self.cat_bottleneck:
                            layers += [
                                nn.LayerNorm(self._hstu_embedding_dim),
                                nn.Linear(self._hstu_embedding_dim, self.cat_bottleneck_dim),
                                nn.SiLU(),
                                nn.Linear(self.cat_bottleneck_dim, self._hstu_embedding_dim),
                            ]
                        layers += [ResBlock(self._hstu_embedding_dim, use_norm=self.head_norm, zero_init=False)
                                for _ in range(medusa_num_layers)]
                        return nn.Sequential(*layers)

                    self.medusa_cat_head = nn.ModuleList([_make_cat_block() for _ in range(self.num_prior_head)])

                    def _make_seg_block():
                        return nn.Sequential(*[
                            ResBlock(self._hstu_embedding_dim, use_norm=self.head_norm, zero_init=False)
                            for _ in range(medusa_num_layers)
                        ])

                    if self.share_seg_weights:
                        shared_seg = _make_seg_block()
                        self.medusa_seg_head = nn.ModuleList([
                            nn.ModuleList([shared_seg for _ in range(self.num_segment_head)])
                            for _ in range(self.num_prior_head)
                        ])
                    else:
                        self.medusa_seg_head = nn.ModuleList([
                            nn.ModuleList([_make_seg_block() for _ in range(self.num_segment_head)])
                            for _ in range(self.num_prior_head)
                        ])
                else:
                    self.medusa_head = nn.ModuleList(
                        [
                            nn.Sequential(
                                *([ResBlock(self._hstu_embedding_dim)] * medusa_num_layers)
                            )
                            for _ in range(self.medusa_num_heads)
                        ]
                    )
                self.weighted_prior_loss = config['weighted_prior_loss']
                if self.loss in ['prior']:
                    if self.category_by == 'item' and not self.neg_sample_by_cat:
                        self.logger.info('===== WARNING =====\n'
                                         'Prior loss works better when we only draw negative '
                                         'samples from the same category')
                else:
                    assert self.num_prior_head == 1, 'Only prior loss is allowed for num_prior_head > 1'

                if self.loss in ['prior'] and self.weighted_prior_loss:
                    all_counts = sum(dataload.category_counts.values())
                    self.prior_loss_weight = [0 for _ in range(self.num_prior_head)]
                    for cat_name, count in dataload.category_counts.items():
                        self.prior_loss_weight[dataload.category_to_int[cat_name]] = count / all_counts
                    self.logger.info(f'Prior weights: {self.prior_loss_weight}, count: {dataload.category_counts}')
                else:
                    self.prior_loss_weight = [1 / self.num_prior_head for _ in range(self.num_prior_head)]

                if self.loss in ['prior'] and config['prior_switch'] is not None:
                    self.use_asym_switch_loss = config.get('asym_switch_loss', False)
                    self.switch_last_only = config.get('switch_last_only', False)
                    if self.use_asym_switch_loss:
                        self.switch_loss = AsymmetricLoss(gamma_pos=config.get('gamma_pos', 4.0), gamma_neg=config.get('gamma_neg', 0.0))
                    else:
                        self.switch_loss = None
                    assert config['split_mode'] == 'combine'
                    self.master_switch = config.get('master_switch', False)
                    if config['prior_switch'] == 'in':
                        self.aux_cat_head = nn.ModuleList(
                            [
                                nn.Linear(self._hstu_embedding_dim, 1)
                                for _ in range(self.num_prior_head)
                            ]
                        )
                    elif config['prior_switch'] == 'in_out':
                        if self.head_interaction == 'multiplicative':
                            assert self.num_segment_head == 1, 'multiplicative head interaction is not supported for prior_switch=in_out when num_segment_head > 1'
                        self.aux_cat_head = nn.ModuleList(
                            [
                                nn.Linear(self._hstu_embedding_dim * 2, 1)
                                for _ in range(self.num_prior_head)
                            ]
                        )
                    else:
                        self.logger.info(f'prior_switch={config["prior_switch"]} is not recognized. Setting to None...')
                        config['prior_switch'] = None
                    if self.master_switch:
                        for i in range(1, self.num_prior_head):
                            for p in self.aux_cat_head[i].parameters():
                                p.requires_grad_(False)
                    self.prior_switch_loss_weight = config['prior_switch_loss_weight']
        self.prior_switch = config['prior_switch']
        self.use_prior_switch_test = config.get('use_prior_switch_test', False)
        self.detach_aux_in = config.get('detach_aux_in', False)
        self.eval_pred_len = config['eval_pred_len']
        self.prior_given_at_test = config.get('prior_given_at_test', False)
        if self.prior_given_at_test:
            self.given_prior_len = config.get('given_prior_len', self.eval_pred_len)
        else:
            self.given_prior_len = self.eval_pred_len
        self.int_to_category = config["int_to_category"]
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
        
        output_embs = cur_embs_valid / cur_embs_valid.norm(dim=-1, keepdim=True)
        target_pos_embs = target_pos_valid / target_pos_valid.norm(dim=-1, keepdim=True) # [batch_size * seq_len, dim]
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
        items, neg_items, user_attention_mask, pos_tag_categories = interaction
        device = user_attention_mask.device
        user_attention_mask = user_attention_mask.bool()
        # items: [b_sz, seq_len+pred_len], neg_items: [b_sz, num_cats, num_negs]),
        # masked_index: [b_sz, seq_len+pred_len], pos_tag_categories: [b_sz, 1 or 0]
        pos_items_embs = self.item_id_proj_tower(self.item_embedding(items))  # [batch, seq_len+pred_len, dim]
        input_emb = pos_items_embs[:, :-self.pred_len, :]  # [batch, seq_len, dim]

        position_ids = torch.arange(input_emb.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(user_attention_mask[:, :-self.pred_len])
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_emb + position_embedding

        attention_mask = self.get_attention_mask(user_attention_mask[:, :-self.pred_len])
        output_embs = self._hstu(x=input_emb, attention_mask=attention_mask)

        model_out = defaultdict(float)
        B = output_embs.shape[0]
        big_batch = B * self.pred_len
        dtype = output_embs.dtype
        if self.head_interaction == 'hierarchical':
            cat_embs = [self.medusa_cat_head[c](output_embs) for c in range(self.num_prior_head)]
            head_embs = output_embs.new_empty((B, self.medusa_num_heads, self.max_seq_length, self._hstu_embedding_dim))  # [B, H, L, D]
            h_ptr = 0
            for s in range(self.num_segment_head):
                seg_bias = self.segment_emb.weight[s] if self.use_seg_embed else None
                for c in range(self.num_prior_head):
                    seg_in = cat_embs[c]
                    if seg_bias is not None:
                        seg_in = seg_in + seg_bias
                    head_embs[:, h_ptr] = self.medusa_seg_head[c][s](seg_in)
                    h_ptr += 1
        else:
            head_embs = torch.stack([
                head(output_embs) for head in self.medusa_head
            ], dim=1)  # [batch, num_heads, seq_len, dim]

        if not self.neg_sample_by_cat or (self.loss == 'prior' and self.head_interaction == 'additive'):
            neg_embedding = self.item_id_proj_tower(self.item_embedding(neg_items[:, -1]))
            D = neg_embedding.size(-1)
            neg_embedding = neg_embedding / neg_embedding.norm(dim=-1, keepdim=True)
            neg_embedding_all = all_gather(neg_embedding, sync_grads=True).reshape(-1, D)  # [total_num_negs, dim]

        if self.loss == 'nce' or (self.loss == 'prior' and self.head_interaction == 'additive'):
            # pick the right head for every prediction offset
            head_for_pred = torch.arange(self.pred_len, device=device) // self.seg_len  # (P,)
            cur_embs = head_embs[:, head_for_pred]  # (B, P, L, D)

            # ─── (2)  FUTURE TARGET WINDOWS & MASKS  (cheap views) ────────────────────────
            # pos_items_embs: (B, L + P, D)   user_attention_mask: (B, L + P)
            windows_pos = pos_items_embs.unfold(1, self.pred_len, 1)[:, 1:].permute(0, 3, 1, 2)  # (B, P, L, D)
            windows_mask = user_attention_mask.unfold(1, self.pred_len, 1)[:, 1:].permute(0, 2, 1)  # (B, P, L)
            base_mask = user_attention_mask[:, :self.max_seq_length].unsqueeze(1)  # (B, 1, L)
            final_mask = base_mask & windows_mask  # (B, P, L)

            # ─── (3)  FLATTEN  (batch × pred_idx)  AXIS  ──────────────────────────────────
            mask_flat = final_mask.reshape(big_batch, self.max_seq_length).bool()
            cur_flat = cur_embs.reshape(big_batch, self.max_seq_length, self._hstu_embedding_dim)[mask_flat]    # (BP, L, D)
            pos_flat = windows_pos.reshape(big_batch, self.max_seq_length, self._hstu_embedding_dim)[mask_flat]

            # ─── (5)  ONE‑SHOT  InfoNCE ───────────────────────────────────────────────────
            logits, labels = self.nce_loss(cur_flat, pos_flat, neg_embedding_all)
            # logits: (N_tok, 1+N_neg)   where N_tok = mask_flat.sum()

            # ─── (6)  TOKEN‑LEVEL CROSS‑ENTROPY  ───────────────────────────────────────────
            tok_loss = F.cross_entropy(logits, labels, reduction='none')   # (N_tok,)

            # map every token back to its prediction offset 0 … P‑1
            pred_idx_token = (torch.arange(self.pred_len, device=device).repeat(B)        # (BP ,)
                            .unsqueeze(1).expand(-1, self.max_seq_length).reshape(-1)[mask_flat.reshape(-1)])

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

            # ─── (8)  SEGMENT (HEAD) LOSSES FOR LOGGING  ───────────────────────────────────
            seg_loss = per_pred_loss.detach().view(self.num_segment_head, self.seg_len).sum(dim=1)  # (H ,)
            for head_idx in range(self.num_segment_head):
                model_out[f"seg_{head_idx}_loss"] = seg_loss[head_idx]

            # ─── (9)  OPTIONAL TOP‑k METRICS  (only pred_idx == 0)  ───────────────────────
            idx0 = pred_idx_token == 0
            if idx0.any():
                model_out.update(self.log_topk_during_train(logits[idx0], labels[idx0]))

        if self.loss == 'prior':
            if self.head_interaction == 'additive':
                seg_len = self.pred_len
            else:
                seg_len = self.seg_len

            if self.prior_switch is not None:
                pos_tag_targets = pos_tag_categories.unfold(dimension=1, size=self.pred_len, step=1)
                # pos_tag_categories: [b_sz, seq_len + pred_len, num_categories]
                pos_tag_targets = pos_tag_targets[:, 1:]
                # pos_tag_targets: [b_sz, seq_len, num_categories, pred_len]

            segment_for_pred = torch.arange(self.pred_len, device=device) // seg_len 
            lam_vec = self.horizon_discount.to(dtype=dtype)
            per_pred_loss_accum = torch.zeros(self.pred_len, device=device, dtype=torch.float32)  # float32 to avoid overflow

            # ─── (2)  FUTURE TARGET WINDOWS & MASKS  (cheap views) ────────────────────────
            # pos_items_embs: (B, L + P, D)   user_attention_mask: (B, L + P)
            windows_pos = pos_items_embs.unfold(1, self.pred_len, 1)[:, 1:].permute(0, 3, 1, 2)  # (B, P, L, D)
            windows_pos = windows_pos.reshape(big_batch, self.max_seq_length, self._hstu_embedding_dim)
            windows_mask = user_attention_mask.unfold(1, self.pred_len, 1)[:, 1:].permute(0, 2, 1)  # (B, P, L)
            base_mask = user_attention_mask[:, :self.max_seq_length].unsqueeze(1) & windows_mask  # (B, 1, L) & (B, P, L) = (B, P, L)

            for prior_idx in range(self.num_prior_head):
                model_out[f'head_nce_{self.int_to_category[prior_idx]}_loss'] = 0

                if self.neg_sample_by_cat:  # only use negative samples from the same category
                    neg_embedding = self.item_id_proj_tower(self.item_embedding(neg_items[:, prior_idx]))
                    D = neg_embedding.size(-1)
                    neg_embedding = neg_embedding / neg_embedding.norm(dim=-1, keepdim=True)
                    neg_embedding_all = all_gather(neg_embedding, sync_grads=True).reshape(-1, D)  # [total_num_negs, dim]

                if self.prior_switch is not None:
                    if self.master_switch and prior_idx > 0:
                        pass
                    else:
                        # pos_tag_targets.shape: [b_sz, seq_len, num_categories, pred_len]
                        if self.switch_last_only:
                            pos_this_tag_targets = torch.any(pos_tag_targets[:, -1:, prior_idx], dim=-1).float()
                        else:
                            pos_this_tag_targets = torch.any(pos_tag_targets[:, :, prior_idx], dim=-1).float()
                        # pos_this_tag_targets.shape [b_sz, seq_len]
                        
                        if self.prior_switch == 'in':
                            aux_in = output_embs.detach() if self.detach_aux_in else output_embs
                            if self.switch_last_only:
                                aux_in = aux_in[:, -1:]
                            pred_prior_logits = self.aux_cat_head[prior_idx](aux_in).squeeze(-1)
                        elif self.prior_switch == 'in_out':  # head_embs: [batch, num_heads, seq_len, dim]
                            if self.head_interaction == 'additive':
                                aux_in = torch.cat([output_embs, head_embs[:, self.num_segment_head + prior_idx]], dim=-1)
                            else:
                                aux_in = torch.cat([output_embs, head_embs[:, prior_idx]], dim=-1)
                            if self.switch_last_only:
                                aux_in = aux_in[:, -1:]
                            if self.detach_aux_in:
                                aux_in = aux_in.detach()
                            pred_prior_logits = self.aux_cat_head[prior_idx](aux_in).squeeze(-1)
                        else:
                            raise ValueError(f'prior_switch={self.prior_switch} not recognized')
                            
                        if self.use_asym_switch_loss:
                            head_cat_loss = self.switch_loss(pred_prior_logits, pos_this_tag_targets)
                        else:
                            p = float(self.prior_loss_weight[prior_idx])  # p is the likelihood of a positive for this head
                            p = max(min(p, 1.0 - 1e-6), 1e-6)

                            pos_w = torch.tensor((1.0 - p) / p, device=pred_prior_logits.device, dtype=pred_prior_logits.dtype)

                            head_cat_loss = F.binary_cross_entropy_with_logits(
                                pred_prior_logits, pos_this_tag_targets, pos_weight=pos_w
                            )

                        head_cat_acc = torch.mean(
                            ((pred_prior_logits >= 0).int() == pos_this_tag_targets.int()).float()
                        )
                        model_out[f'head_cat_{self.int_to_category[prior_idx]}_acc'] = head_cat_acc.detach()
                        model_out['loss'] += self.prior_switch_loss_weight * head_cat_loss
                        model_out[f'head_cat_{self.int_to_category[prior_idx]}_loss'] = (
                            self.prior_switch_loss_weight * head_cat_loss.detach()
                        )

                # category mask for positives (with optional mix-in)
                prior_full = pos_tag_categories[:, :, prior_idx]  # (B, L+P)
                prior_win  = prior_full.unfold(1, self.pred_len, 1)[:, 1:].permute(0, 2, 1)  # (B, P, L)
                if self.pos_sample_mix_ratio > 0.0:
                    mix_mask = torch.rand_like(prior_win, dtype=torch.float) < self.pos_sample_mix_ratio
                    prior_win = prior_win | mix_mask
                final_mask = base_mask & prior_win  # (B, P, L)
                mask_flat = final_mask.reshape(big_batch, self.max_seq_length).bool()  # flatten (batch × pred_idx) axis
                empty_mask = mask_flat.sum() == 0
                if empty_mask:
                    k = max(1, int(0.10 * mask_flat.numel()))
                    rand_idx = torch.randperm(mask_flat.numel(), device=mask_flat.device)[:k]
                    mask_flat.view(-1)[rand_idx] = True    

                # pick the right head for every prediction offset
                if self.head_interaction == 'additive':
                    head_for_pred = torch.full((self.pred_len,), self.num_segment_head + prior_idx, device=device, dtype=torch.long)
                else:
                    head_for_pred = segment_for_pred * self.num_prior_head + prior_idx

                cur_embs = head_embs[:, head_for_pred]  # (B, P, L, D)
                cur_flat = cur_embs.reshape(big_batch, self.max_seq_length, self._hstu_embedding_dim)[mask_flat]    # (BP, L, D)
                pos_flat = windows_pos.reshape(big_batch, self.max_seq_length, self._hstu_embedding_dim)[mask_flat]
                
                if not empty_mask:
                    logits, labels = self.nce_loss(cur_flat, pos_flat, neg_embedding_all)
                    tok_loss = F.cross_entropy(logits, labels, reduction='none')
                else:  # all-masked safety guard
                    print(f'Warning, rank={self.rank} has all zeros, loss set to zero')
                    logits, labels = self.nce_loss(cur_flat, pos_flat, neg_embedding_all)
                    tok_loss = F.cross_entropy(logits, labels, reduction='none')
                    tok_loss = (tok_loss * torch.zeros_like(tok_loss)).mean()
                    continue

                # map every token back to its prediction offset 0 … P‑1
                pred_idx_token = (torch.arange(self.pred_len, device=device).repeat(B)  # (BP ,)
                                .unsqueeze(1).expand(-1, self.max_seq_length).reshape(-1)[mask_flat.reshape(-1)])

                # sum & count per offset in O(N) with one index_add_ each
                loss_sum_per_p = torch.zeros(self.pred_len, device=device, dtype=torch.float32)
                cnt_per_p = torch.zeros_like(loss_sum_per_p)
                loss_sum_per_p.index_add_(0, pred_idx_token, tok_loss.float())
                cnt_per_p.index_add_(0, pred_idx_token, torch.ones_like(tok_loss, dtype=torch.float32))
                mean_loss_per_p = (loss_sum_per_p / torch.clamp_min(cnt_per_p, 1.0)).to(dtype=dtype)

                per_pred_loss = lam_vec * self.prior_loss_weight[prior_idx] * mean_loss_per_p  # (P,)
                model_out["loss"] += per_pred_loss.sum()
                per_pred_loss_accum += per_pred_loss.to(per_pred_loss_accum.dtype)

                # logging like original code
                cat_name = self.int_to_category[prior_idx]
                model_out[f"head_nce_{cat_name}_loss"] = per_pred_loss.sum().detach()

                # optional top-k only for first prior & first pred-idx
                if prior_idx == 0 and (pred_idx_token == 0).any():
                    idx0 = pred_idx_token == 0
                    model_out.update(self.log_topk_during_train(logits[idx0], labels[idx0]))

            seg_loss = per_pred_loss_accum.detach().view(self.num_segment_head, self.seg_len).sum(dim=1)   # (H_total ,)
            if self.head_interaction != 'additive':
                for head_idx in range(self.num_segment_head):
                    model_out[f"seg_{head_idx}_loss"] += seg_loss[head_idx]
            else:
                model_out['loss'] = model_out['loss'] / 2

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

        pred_prior_cats = []
        if 1 == 0:  # or self.medusa_head is None:
            generated_embeddings = []
            for head_idx in range(self.medusa_num_heads):
                attention_mask = self.get_attention_mask(item_seq)
                output_embs = self._hstu(
                    x=item_emb,
                    attention_mask=attention_mask
                )
                next_token_embedding = output_embs[:, -1:]
                next_token_embedding = next_token_embedding / next_token_embedding.norm(dim=-1, keepdim=True)

                # Append the new token embedding to the generated sequence
                generated_embeddings.append(next_token_embedding)
                item_emb = torch.cat([item_emb, next_token_embedding], dim=1)
                item_seq = torch.cat([
                    item_seq, torch.ones((item_seq.shape[0], 1), dtype=item_seq.dtype, device=item_seq.device)], dim=1)
            
            # Concatenate all generated embeddings along with the original sequence output
            full_seq_output = torch.cat(generated_embeddings, dim=1)

        else:
            attention_mask = self.get_attention_mask(item_seq)
            output_embs = self._hstu(
                x=item_emb,
                attention_mask=attention_mask
            )
            next_token_embedding = output_embs[:, -1]

            if self.head_interaction == 'hierarchical':
                cat_embs = [self.medusa_cat_head[c](next_token_embedding) for c in range(self.num_prior_head)]
                full_seq_output = output_embs.new_empty((output_embs.shape[0], self.medusa_num_heads, self._hstu_embedding_dim))  # [B, H, D]
                h_ptr = 0
                for s in range(self.num_segment_head):
                    seg_bias = self.segment_emb.weight[s] if self.use_seg_embed else None
                    for c in range(self.num_prior_head):
                        seg_in = cat_embs[c]
                        if seg_bias is not None:
                            seg_in = seg_in + seg_bias
                        full_seq_output[:, h_ptr] = self.medusa_seg_head[c][s](seg_in)
                        h_ptr += 1
            else:
                full_seq_output= torch.stack(
                    [h(next_token_embedding) for h in self.medusa_head],
                    dim=1,                                                # (B, num_heads, D)
                )

            wandb_logs['num_samples'] = self.eval_pred_len * output_embs.shape[0]

            if self.prior_switch is not None:
                if self.master_switch:
                    switch_range = 1
                else:
                    switch_range = self.num_prior_head
                for head_idx in range(switch_range):
                    if self.prior_switch == 'in':
                        pred_prior_logits = self.aux_cat_head[head_idx](next_token_embedding)
                    elif self.prior_switch == 'in_out':
                        if self.head_interaction == 'additive':
                            aux_in = torch.cat([next_token_embedding, full_seq_output[:, self.num_segment_head + head_idx]], dim=-1)
                        else:
                            aux_in = torch.cat([next_token_embedding, full_seq_output[:, head_idx]], dim=-1)
                        pred_prior_logits = self.aux_cat_head[head_idx](aux_in)
                    else:
                        raise ValueError(f'prior_switch={self.prior_switch} not recognized')

                    pred_bin = (pred_prior_logits >= 0).to(torch.bool).squeeze(-1)
                    pred_prior_cats.append(pred_bin)
                    prior_cat_labels = (torch.sum(target_tags[:, :, head_idx], dim=-1) > 0).squeeze(-1).to(torch.bool)
                    wandb_logs[f'head_cat_{self.int_to_category[head_idx]}_num_correct'] = \
                        torch.sum((prior_cat_labels == pred_prior_cats[head_idx]) * 1.0)

        if save_for_eval:
            saved_user_embs = (next_token_embedding / next_token_embedding.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        else:
            saved_user_embs = None
       
        del next_token_embedding, output_embs
        # full_seq_output shape: (b_sz, num_heads, d_model)
        full_seq_output = full_seq_output.float()
        final_seq_output = full_seq_output / full_seq_output.norm(dim=-1, keepdim=True)  # Normalize

        if save_for_eval:
            saved_head_embs = final_seq_output.float().cpu().numpy()
        else:
            saved_head_embs = None

        # Normalize item_feature for cosine similarity
        all_item_feature = all_item_feature.float()
        all_item_feature = all_item_feature / all_item_feature.norm(dim=-1, keepdim=True)
        # all_item_feature.shape: [all_candidate_items, d_model]

        # Perform similarity calculation with each generated token
        similarity_scores = torch.matmul(final_seq_output, all_item_feature.t())
        # similarity_scores shape: (b_sz, num_heads, all_items)
        # target_tags: [b_sz, pred_len, num_categories]
        if self.loss == 'prior':
            if self.prior_given_at_test:
                target_tags_mask = target_tags[:, :self.given_prior_len].bool().any(dim=1)
                if self.head_interaction == 'additive':
                    target_tags_mask = target_tags_mask.unsqueeze(-1)
                    similarity_scores[:, self.num_segment_head:].masked_fill_(~target_tags_mask, float('-inf'))
                else:
                    target_tags_mask = target_tags_mask.repeat(1, self.num_segment_head).unsqueeze(-1)
                    similarity_scores.masked_fill_(~target_tags_mask, float('-inf'))

            # selected items must come from the corresponding head
            # all_item_tags: [all_candidate_items, num_categories]
            if self.head_interaction == 'additive':
                all_item_tags = all_item_tags.bool().unsqueeze(0)
                similarity_scores[:, self.num_segment_head:].masked_fill_(~all_item_tags, float('-inf'))
            else:
                all_item_tags = all_item_tags.bool().repeat(self.num_segment_head, 1).unsqueeze(0)
                similarity_scores.masked_fill_(~all_item_tags, float('-inf'))

            # turn off heads according to prior switch
            if self.prior_switch is not None and self.use_prior_switch_test:
                if self.master_switch:
                    pred_prior_mask = torch.zeros(similarity_scores.shape[0], self.num_prior_head, 1, dtype=torch.bool, device=similarity_scores.device)
                    pred_prior_mask[:, 0, 0] = ~pred_prior_cats[0]
                    pred_prior_mask[:, 1:, 0] = pred_prior_cats[0].unsqueeze(-1)
                else:
                    # pred_prior_cats: list of [b_sz], len = num_prior_heads
                    pred_prior_mask = ~torch.stack([t.view(-1, 1) for t in pred_prior_cats], dim=1).to(torch.bool)
                # pred_prior_cats.shape: [b_sz, num_prior_heads, 1]
                if self.head_interaction == 'additive':
                    similarity_scores[:, self.num_segment_head:].masked_fill_(pred_prior_mask, float('-inf'))
                else:
                    mask_all = pred_prior_mask.repeat(1, self.num_segment_head, 1)
                    similarity_scores.masked_fill_(mask_all, float('-inf'))
        return similarity_scores, wandb_logs, saved_user_embs, saved_head_embs

    @torch.no_grad()
    def compute_item_all(self):
        weight = self.item_id_proj_tower(self.item_embedding.weight)
        return weight / weight.norm(dim=-1, keepdim=True)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        # extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask
