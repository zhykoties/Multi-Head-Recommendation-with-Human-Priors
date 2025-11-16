from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from logging import getLogger
from typing import Optional

from REC.llm_dict import use_image_dict
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel, all_gather
from REC.model.llm_heads import ResBlock, Rescale
from REC.model.HLLM.modeling_llama import LlamaForCausalLM
from REC.model.HLLM.modeling_mistral import MistralForCausalLM
from REC.model.HLLM.modeling_bert import BertModel
from REC.model.HLLM.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from REC.model.HLLM.modeling_qwen2 import Qwen2ForCausalLM
from REC.model.HLLM.baichuan.modeling_baichuan import BaichuanForCausalLM
from REC.model.HLLM.modeling_llava_onevision import LlavaOnevisionForConditionalGeneration
from REC.model.HLLM.modeling_llava_next import LlavaNextForConditionalGeneration

from REC.model.layers import DummyLLM, AsymmetricLoss


def random_reorder(tensor):
    """
    [Testing Only] To validate whether the model is sensitive to the input sequence order.
    """
    batch_size, num_items = tensor.shape
    # Generate random permutations for each batch
    permutations = torch.stack([torch.randperm(num_items) for _ in range(batch_size)])
    # Create an index tensor for advanced indexing
    batch_indices = torch.arange(batch_size).unsqueeze(1).to(tensor.device)  # Shape: [batch_size, 1]
    return tensor[batch_indices, permutations.to(tensor.device)]


class HLLM(BaseModel):
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(HLLM, self).__init__()
        self.logger = getLogger()
        self.dummy_llm = config['dummy_llm']
        self.freeze_item_llm = config.get('freeze_item_llm', False)
        self.all_item_embeds = None

        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.pred_len = config['pred_len']

        self.item_pretrain_dir = config['item_pretrain_dir']
        self.user_pretrain_dir = config['user_pretrain_dir']
        self.gradient_checkpointing = config['gradient_checkpointing']
        self.use_ft_flash_attn = config['use_ft_flash_attn']
        self.logger.info(f"create item llm")
        self.item_llm = self.create_llm(self.item_pretrain_dir, config['item_llm_init'])
        self.logger.info(f"create user llm")
        self.user_llm = self.create_llm(self.user_pretrain_dir, config['user_llm_init'])
        self.item_emb_token_n = config['item_emb_token_n']  # has to be 1 or None (using mean-pooling)
        idx = config['item_pretrain_dir'].rfind('/')
        if idx == -1:
            self.message_type = use_image_dict[config['item_pretrain_dir']]['message_type']
        else:
            self.message_type = use_image_dict[config['item_pretrain_dir'][idx + 1:]]['message_type']
        self._tweak_checkpointing_for_frozen_parts()

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

        if self.item_emb_token_n > 1:
            raise NotImplementedError(f"Not support item_emb_token_n {self.item_emb_token_n} > 1")

        if self.dummy_llm:
            self.item_llm_hidden_size = self.item_llm.hidden_size
            self.user_llm_hidden_size = self.user_llm.hidden_size
        elif hasattr(self.item_llm.config, "hidden_size"):
            self.item_llm_hidden_size = self.item_llm.config.hidden_size
            self.user_llm_hidden_size = self.user_llm.config.hidden_size
        else:
            self.item_llm_hidden_size = self.item_llm.config.text_config.hidden_size
            self.user_llm_hidden_size = self.user_llm.config.text_config.hidden_size

        if self.item_emb_token_n > 0:
            self.item_emb_tokens = nn.Parameter(
                torch.zeros(1, self.item_emb_token_n, self.item_llm_hidden_size)
            )
            self.item_emb_tokens.data.normal_(mean=0.0, std=0.02)
            if config['item_emb_pretrain']:
                ckpt = torch.load(config['item_emb_pretrain'], map_location='cpu')
                self.logger.info(f"load item_emb_token from {config['item_emb_pretrain']} with {ckpt.size()}")
                self.item_emb_tokens.data = nn.Parameter(ckpt)
        else:  # mean pooling
            self.item_emb_tokens = None

        self.loss = config['loss']
        self.neg_sample_by_cat = config['neg_sample_by_cat']
        if self.loss != 'prior':
            self.neg_sample_by_cat = False
        self.pos_sample_mix_ratio = config['pos_sample_mix_ratio']
        if self.loss in ['nce', 'prior']:
            if config['fix_temp']:
                self.logger.info(f"Fixed logit_scale 1/0.07")
                self.register_buffer("logit_scale", torch.tensor(np.log(1 / 0.07)))
            else:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
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
                    self.cat_bottleneck_dim = config.get("cat_bottleneck_dim", self.user_llm_hidden_size // 2)
                    self.share_seg_weights  = config.get("share_seg_weights", False)  # tie segment heads per category
                    self.use_seg_embed      = config.get("segment_embed", False)       # learned segment embedding

                    if self.use_seg_embed:
                        self.segment_emb = nn.Embedding(self.num_segment_head, self.user_llm_hidden_size)

                    def _make_cat_block():
                        layers = []
                        if self.cat_bottleneck:
                            layers += [
                                nn.LayerNorm(self.user_llm_hidden_size),
                                nn.Linear(self.user_llm_hidden_size, self.cat_bottleneck_dim),
                                nn.SiLU(),
                                nn.Linear(self.cat_bottleneck_dim, self.user_llm_hidden_size),
                            ]
                        layers += [ResBlock(self.user_llm_hidden_size, use_norm=self.head_norm, zero_init=False)
                                for _ in range(medusa_num_layers)]
                        return nn.Sequential(*layers)

                    self.medusa_cat_head = nn.ModuleList([_make_cat_block() for _ in range(self.num_prior_head)])

                    def _make_seg_block():
                        return nn.Sequential(*[
                            ResBlock(self.user_llm_hidden_size, use_norm=self.head_norm, zero_init=False)
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
                                *([ResBlock(self.user_llm_hidden_size)] * medusa_num_layers)
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
                                nn.Linear(self.user_llm_hidden_size, 1)
                                for _ in range(self.num_prior_head)
                            ]
                        )
                    elif config['prior_switch'] == 'in_out':
                        if self.head_interaction == 'multiplicative':
                            assert self.num_segment_head == 1, 'multiplicative head interaction is not supported for prior_switch=in_out when num_segment_head > 1'
                        self.aux_cat_head = nn.ModuleList(
                            [
                                nn.Linear(self.user_llm_hidden_size * 2, 1)
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

    def _disable_ckpt_on_module(self, module, tag=""):
        # Try the common switches used by HF stacks
        try:
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()
        except Exception:
            pass
        try:
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
        except Exception:
            pass
        try:
            if hasattr(module, "config"):
                if hasattr(module.config, "gradient_checkpointing"):
                    module.config.gradient_checkpointing = False
                # Optional: cache can be re-enabled for frozen/no-grad towers
                if hasattr(module.config, "use_cache"):
                    module.config.use_cache = True
        except Exception:
            pass
        # Some models put the flag on children too; turn it off recursively
        for child in module.modules():
            if child is not module and hasattr(child, "gradient_checkpointing"):
                try:
                    child.gradient_checkpointing = False
                except Exception:
                    pass

    def _tweak_checkpointing_for_frozen_parts(self):
        # 1) Visual tower is always run under no_grad in your code → disable ckpt there
        vis = getattr(self.item_llm, "visual", None)
        if vis is not None:
            self._disable_ckpt_on_module(vis, "item_llm.visual")

        # 2) If you freeze the entire item LLM, also disable ckpt on its text stack
        if getattr(self, "freeze_item_llm", False):
            self._disable_ckpt_on_module(self.item_llm, "item_llm")

    def create_llm(self, pretrain_dir, init=True):
        self.logger.info(f"******* create LLM {pretrain_dir} *******")
        hf_config = AutoConfig.from_pretrained(pretrain_dir, trust_remote_code=True)
        self.logger.info(f"hf_config: {hf_config}")
        hf_config.gradient_checkpointing = self.gradient_checkpointing
        hf_config.use_cache = False
        hf_config.output_hidden_states = True
        hf_config.return_dict = True
        if self.dummy_llm:
            self.logger.info('Using a dummy LLM for debugging...')
            vocab_size = getattr(hf_config, 'vocab_size', None)
            hidden_size = getattr(hf_config, 'hidden_size', None)
            return DummyLLM(vocab_size, hidden_size)

        self.logger.info("xxxxx starting loading checkpoint")
        if isinstance(hf_config, transformers.LlamaConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for llama')
            self.logger.info(f'Init {init} for llama')
            if init:
                return LlamaForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return LlamaForCausalLM(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.MistralConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for mistral')
            self.logger.info(f'Init {init} for mistral')
            if init:
                return MistralForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return MistralForCausalLM(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.BertConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for bert')
            self.logger.info(f'Init {init} for bert')
            if init:
                return BertModel.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return BertModel(config=hf_config).cuda()
        elif getattr(hf_config, "model_type", None) == "baichuan":
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for baichuan')
            self.logger.info(f'Init {init} for baichuan')
            if init:
                return BaichuanForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return BaichuanForCausalLM(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.Qwen2VLConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for qwen2vl')
            self.logger.info(f'Init {init} for qwen2vl')
            if init:
                return Qwen2VLForConditionalGeneration.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return Qwen2VLForConditionalGeneration(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.Qwen2Config):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for qwen2')
            self.logger.info(f'Init {init} for qwen2')
            if init:
                return Qwen2ForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return Qwen2ForCausalLM(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.LlavaOnevisionConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for Llava One Vision')
            self.logger.info(f'Init {init} for Llava One Vision')
            if init:
                return LlavaOnevisionForConditionalGeneration.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return LlavaOnevisionForConditionalGeneration(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.LlavaNextConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for LlavaNext')
            self.logger.info(f'Init {init} for LlavaNext')
            if init:
                return LlavaNextForConditionalGeneration.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return LlavaNextForConditionalGeneration(config=hf_config).cuda()
        else:
            return AutoModelForCausalLM.from_pretrained(
                self.local_dir, config=hf_config
            )

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

    def forward_item_emb(
        self,
        input_ids,
        position_ids,
        cu_input_lens,
        emb_token_n,
        emb_tokens,
        llm,  # item_llm
        pixel_values: Optional[torch.tensor] = None,
        image_grid_thw: Optional[torch.tensor] = None
    ):
        emb_pos = cu_input_lens.cumsum(dim=0, dtype=torch.int32)
        if pixel_values.size() == torch.Size([0]):  # no image
            model_out = llm(
                input_ids=input_ids,
                position_ids=position_ids.unsqueeze(0),
                cu_input_lens=cu_input_lens,
                emb_tokens=emb_tokens,
                emb_pos=emb_pos,
                emb_token_n=emb_token_n
            )
        else:
            if self.message_type in ['llama']:
                model_out = llm(
                    input_ids=input_ids,
                    position_ids=position_ids.unsqueeze(0),
                    cu_input_lens=cu_input_lens,
                    pixel_values=pixel_values,
                    image_sizes=image_grid_thw,
                    emb_tokens=emb_tokens,
                    emb_pos=emb_pos,
                    emb_token_n=emb_token_n
                )
            elif self.message_type in ['qwen']:
                model_out = llm(
                    input_ids=input_ids,
                    position_ids=position_ids.unsqueeze(0),
                    cu_input_lens=cu_input_lens,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    emb_tokens=emb_tokens,
                    emb_pos=emb_pos,
                    emb_token_n=emb_token_n
                )
        model_out = model_out.hidden_states[-1].squeeze(0)

        if self.dummy_llm:
            emb = model_out[:cu_input_lens.size(0)]
        elif emb_token_n > 0:
            emb = model_out[emb_pos - 1]
        else:  # mean pooling
            max_len = cu_input_lens.max().item()
            cu_seqlens = F.pad(cu_input_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
            seqs = [model_out[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
            padded_seqs = [
                F.pad(
                    seqs[i],
                    (0, 0) * (seqs[i].dim() - 1) + (0, max_len - cu_input_lens[i]),
                    value=0.0,
                )
                for i in range(cu_input_lens.size(0))
            ]
            out = torch.stack(padded_seqs)
            emb = out.sum(dim=1) / cu_input_lens.unsqueeze(1)

        return emb

    def log_topk_during_train(self, logits, labels):
        log_dict = {'nce_samples': (logits > torch.finfo(logits.dtype).min / 100).sum(
            dim=1).float().mean().detach()}
        for k in [1, 5, 10, 50, 100]:
            if k > logits.size(-1):
                break
            indices = logits.topk(k, dim=-1).indices
            log_dict[f'nce_top{k}_acc'] = labels.view(-1, 1).eq(indices).any(dim=-1).float().mean().detach()
        return log_dict

    def forward(self, interaction, mode='train'):
        if mode == 'predict':
            return self.predict(*interaction)
        if mode == 'compute_item':
            return self.compute_item(interaction)
        # which items are padded, padded items always in front for context, at the end for predictions.
        user_attention_mask = interaction['attention_mask'].bool()
        device = user_attention_mask.device
        N, S = user_attention_mask.shape
        if self.freeze_item_llm:
            pos_item_ids = interaction['pos_item_ids']
            pos_items_embs = self.all_item_embeds[pos_item_ids]
            pos_items_embs = pos_items_embs.to(dtype=next(self.user_llm.parameters()).dtype)
            # pos_items_embs shape: [b_sz, seq_len + pred_len, hidden_dim]
        else:
            pos_input_ids, pos_cu_input_lens, pos_position_ids, pos_pixel_values, pos_image_grid_thw = interaction[
                'pos_input_ids'], interaction['pos_cu_input_lens'], interaction['pos_position_ids'], interaction[
                'pos_pixel_values'], interaction['pos_image_grid_thw']

            pos_items_embs = self.forward_item_emb(pos_input_ids, pos_position_ids, pos_cu_input_lens,
                                                  self.item_emb_token_n, self.item_emb_tokens, self.item_llm,
                                                  pos_pixel_values, pos_image_grid_thw)
            pos_items_embs = pos_items_embs.reshape(N, S, -1)
            # pos_items_embs.shape: [b_sz, seq_len + pred_len, d_model]

        user_embedding = self.user_llm(inputs_embeds=pos_items_embs[:, :-self.pred_len],
                                       attention_mask=user_attention_mask[:, :-self.pred_len]).hidden_states[-1]  # last hidden layer

        # user_embedding.shape: [b_sz, seq_len + pred_len, d_model]

        model_out = defaultdict(float)
        B = user_embedding.shape[0]
        big_batch = B * self.pred_len
        dtype = user_embedding.dtype
        if self.head_interaction == 'hierarchical':
            cat_embs = [self.medusa_cat_head[c](user_embedding) for c in range(self.num_prior_head)]
            head_embs = user_embedding.new_empty((B, self.medusa_num_heads, self.max_seq_length, self.user_llm_hidden_size))  # [B, H, L, D]
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
                head(user_embedding) for head in self.medusa_head
            ], dim=1)  # [batch, num_heads, seq_len, dim]

        if not self.neg_sample_by_cat or (self.loss == 'prior' and self.head_interaction == 'additive'):
            if self.freeze_item_llm:
                neg_item_ids = interaction['neg_item_ids']
                neg_embedding = self.all_item_embeds[neg_item_ids]
                neg_embedding = neg_embedding.to(dtype=user_embedding.dtype)
                # neg_embedding shape: [b_sz, seq_len, hidden_dim]
            else:
                neg_input_ids, neg_cu_input_lens, neg_position_ids, neg_pixel_values, neg_image_grid_thw = interaction[
                    'neg_input_ids'], interaction['neg_cu_input_lens'], interaction['neg_position_ids'], interaction[
                    'neg_pixel_values'], interaction['neg_image_grid_thw']

                neg_embedding = self.forward_item_emb(neg_input_ids, neg_position_ids, neg_cu_input_lens,
                                                      self.item_emb_token_n, self.item_emb_tokens, self.item_llm,
                                                      neg_pixel_values, neg_image_grid_thw)
                # neg_embedding.shape: torch.Size([seq_len + pred_len * b_sz, d_model])
                neg_embedding = neg_embedding.reshape(N, -1, self.item_llm_hidden_size)
                # neg_embedding.shape: [b_sz, seq_len + pred_len, d_model]

            neg_embedding = neg_embedding / neg_embedding.norm(dim=-1, keepdim=True)
            D = neg_embedding.size(-1)
            neg_embedding_all = all_gather(neg_embedding, sync_grads=True).reshape(-1, D)  # [num, dim]

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
            cur_flat = cur_embs.reshape(big_batch, self.max_seq_length, self.user_llm_hidden_size)[mask_flat]    # (BP, L, D)
            pos_flat = windows_pos.reshape(big_batch, self.max_seq_length, self.user_llm_hidden_size)[mask_flat]

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

            pos_tag_categories = interaction['pos_tag_categories']
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
            windows_pos = windows_pos.reshape(big_batch, self.max_seq_length, self.user_llm_hidden_size)
            windows_mask = user_attention_mask.unfold(1, self.pred_len, 1)[:, 1:].permute(0, 2, 1)  # (B, P, L)
            base_mask = user_attention_mask[:, :self.max_seq_length].unsqueeze(1) & windows_mask  # (B, 1, L) & (B, P, L) = (B, P, L)

            for prior_idx in range(self.num_prior_head):
                model_out[f'head_nce_{self.int_to_category[prior_idx]}_loss'] = 0

                if self.neg_sample_by_cat:
                    if self.freeze_item_llm:
                        neg_item_ids = interaction[f'neg_item_ids_cat{prior_idx}']
                        neg_embedding = self.all_item_embeds[neg_item_ids]
                        neg_embedding = neg_embedding.to(dtype=user_embedding.dtype)
                        self.logger.info(f'freeze neg_embedding cat={prior_idx}: {neg_embedding.shape}')
                    else:
                        neg_input_ids, neg_cu_input_lens, neg_position_ids, neg_pixel_values, neg_image_grid_thw = interaction[
                            f'neg_input_ids_cat{prior_idx}'], interaction[f'neg_cu_input_lens_cat{prior_idx}'], interaction[
                            f'neg_position_ids_cat{prior_idx}'], interaction[f'neg_pixel_values_cat{prior_idx}'], interaction[
                            f'neg_image_grid_thw_cat{prior_idx}']

                        neg_embedding = self.forward_item_emb(neg_input_ids, neg_position_ids, neg_cu_input_lens,
                                                            self.item_emb_token_n, self.item_emb_tokens, self.item_llm,
                                                            neg_pixel_values, neg_image_grid_thw)
                        # neg_embedding.shape: torch.Size([seq_len + pred_len * b_sz, d_model])
                        neg_embedding = neg_embedding.reshape(N, -1, self.item_llm_hidden_size)
                        # neg_embedding.shape: [b_sz, seq_len + pred_len, d_model]
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
                            aux_in = user_embedding.detach() if self.detach_aux_in else user_embedding
                            if self.switch_last_only:
                                aux_in = aux_in[:, -1:]
                            pred_prior_logits = self.aux_cat_head[prior_idx](aux_in).squeeze(-1)
                        elif self.prior_switch == 'in_out':  # head_embs: [batch, num_heads, seq_len, dim]
                            if self.head_interaction == 'additive':
                                aux_in = torch.cat([user_embedding, head_embs[:, self.num_segment_head + prior_idx]], dim=-1)
                            else:
                                aux_in = torch.cat([user_embedding, head_embs[:, prior_idx]], dim=-1)
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
                cur_flat = cur_embs.reshape(big_batch, self.max_seq_length, self.user_llm_hidden_size)[mask_flat]    # (BP, L, D)
                pos_flat = windows_pos.reshape(big_batch, self.max_seq_length, self.user_llm_hidden_size)[mask_flat]

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
                pred_idx_token = (torch.arange(self.pred_len, device=device).repeat(B)        # (BP ,)
                                .unsqueeze(1).expand(-1, self.max_seq_length).reshape(-1)[mask_flat.reshape(-1)])

                # sum & count per offset in O(N) with one index_add_ each
                loss_sum_per_p = torch.zeros(self.pred_len, device=device, dtype=torch.float32)
                cnt_per_p = torch.zeros_like(loss_sum_per_p)
                loss_sum_per_p.index_add_(0, pred_idx_token, tok_loss.float())
                cnt_per_p.index_add_(0, pred_idx_token, torch.ones_like(tok_loss, dtype=torch.float32))
                mean_loss_per_p = (loss_sum_per_p / torch.clamp_min(cnt_per_p, 1.0)).to(dtype=dtype)

                per_pred_loss = lam_vec * self.prior_loss_weight[prior_idx] * mean_loss_per_p                       # (P,)
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

        # Prepare initial attention mask and item embeddings
        attention_mask = (item_seq > 0).int()

        pos_embedding = all_item_feature[item_seq]
        tokens_per_item = 1

        generated_embeddings = []
        pred_prior_cats = []
        # attention_mask.shape: [b_sz, seq_len]
        # pos_embedding.shape: [b_sz, seq_len, d_model]
        
        output = self.user_llm(inputs_embeds=pos_embedding, attention_mask=attention_mask)
        user_embedding = output.hidden_states[-1]  # Last layer hidden states
        next_token_embedding = user_embedding[:, -1]  # Get the next token embedding

        if self.head_interaction == 'hierarchical':
            cat_embs = [self.medusa_cat_head[c](next_token_embedding) for c in range(self.num_prior_head)]
            full_seq_output = user_embedding.new_empty((user_embedding.shape[0], self.medusa_num_heads, self.user_llm_hidden_size))  # [B, H, D]
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

        wandb_logs['num_samples'] = self.eval_pred_len * user_embedding.shape[0]
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

                # prior for this head (likelihood of positive)
                pred_bin = (pred_prior_logits >= 0).to(torch.bool).squeeze(-1)
                pred_prior_cats.append(pred_bin)
                prior_cat_labels = (torch.sum(target_tags[:, :, head_idx], dim=-1) > 0).squeeze(-1).to(torch.bool)
                wandb_logs[f'head_cat_{self.int_to_category[head_idx]}_num_correct'] = \
                    torch.sum((prior_cat_labels == pred_prior_cats[head_idx]) * 1.0)

        if save_for_eval:
            saved_user_embs = next_token_embedding.float().cpu().numpy()
            saved_head_embs = full_seq_output.float().cpu().numpy()
        else:
            saved_user_embs = None
            saved_head_embs = None

        del next_token_embedding, user_embedding
        # full_seq_output shape: (b_sz, num_heads, d_model)
        full_seq_output = full_seq_output.float()
        final_seq_output = full_seq_output / full_seq_output.norm(dim=-1, keepdim=True)  # Normalize

        # Normalize item_feature for cosine similarity
        all_item_feature = all_item_feature.float()
        all_item_feature = all_item_feature / all_item_feature.norm(dim=-1, keepdim=True)
        # all_item_feature.shape: [all_candidate_items, d_model]

        # Perform similarity calculation with each generated token
        similarity_scores = torch.matmul(final_seq_output, all_item_feature.t())
        # similarity_scores.shape: [b_sz, num_heads, all_candidate_items]
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
        return self.all_item_embeds

    @torch.no_grad()
    def set_all_item_embeds(self, all_item_embeds):
        self.all_item_embeds = all_item_embeds.detach()

    @torch.no_grad()
    def compute_item(self, interaction):
        """
        Given user history, compute an item embedding
        """
        pos_input_ids, pos_cu_input_lens, pos_position_ids, pos_pixel_values, pos_image_grid_thw = interaction[
            'pos_input_ids'], interaction['pos_cu_input_lens'], interaction['pos_position_ids'], interaction[
            'pos_pixel_values'], interaction['pos_image_grid_thw']
        pos_tag_categories = interaction['pos_tag_categories']
        pos_original_tag_categories = interaction['pos_original_tag_categories']
        pos_embedding = self.forward_item_emb(pos_input_ids, pos_position_ids, pos_cu_input_lens, self.item_emb_token_n,
                                              self.item_emb_tokens, self.item_llm, pos_pixel_values, pos_image_grid_thw)
        N = pos_cu_input_lens.size(0)
        pos_embedding = pos_embedding.view(N, -1)

        return pos_embedding, pos_tag_categories, pos_original_tag_categories
