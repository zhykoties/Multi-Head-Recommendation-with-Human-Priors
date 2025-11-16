# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT

from .register import Register
import torch
import copy
import numpy as np

import pandas as pd
import torch


class DataStruct(object):

    def __init__(self):
        self._tensor_lists = {}
        self._data_dict = {}

    def __getitem__(self, name: str):
        return self._data_dict[name]

    def __setitem__(self, name: str, value):
        self._data_dict[name] = value

    def __delitem__(self, name: str):
        self._data_dict.pop(name)

    def __contains__(self, key: str):
        return key in self._data_dict

    def get(self, name: str):
        if name not in self._data_dict:
            raise IndexError("Can not load the data without registration !")
        return self[name]

    def set(self, name: str, value):
        self._data_dict[name] = value

    def update_tensor(self, name: str, value: torch.Tensor):
        if name not in self._tensor_lists:
            self._tensor_lists[name] = []
        self._tensor_lists[name].append(value.cpu().clone().detach())

    def finalize_tensors(self):
        """Call this once at the end to concatenate all accumulated tensors"""
        for name, tensor_list in self._tensor_lists.items():
            if tensor_list:
                self._data_dict[name] = torch.cat(tensor_list, dim=0)
        self._tensor_lists.clear()

    def __str__(self):
        data_info = '\nContaining:\n'
        for data_key in self._data_dict.keys():
            data_info += data_key + '\n'
        return data_info


class Collector(object):
    """The collector is used to collect the resource for evaluator.
        As the evaluation metrics are various, the needed resource not only contain the recommended result
        but also other resource from data and model. They all can be collected by the collector during the training
        and evaluation process.

        This class is only used in Trainer.

    """

    def __init__(self, config):
        self.count = 999
        self.config = config
        self.metrics_pred_len_list = config['metrics_pred_len_list']
        self.eval_pred_len = config['eval_pred_len']
        self.data_struct = dict()
        for pred_idx in self.metrics_pred_len_list:
             self.data_struct[pred_idx] = DataStruct()
        self.data_struct[-1] = DataStruct()  # for common attributes that are shared across pred_len
        self.register = Register(config)
        self.full = True
        self.topk = self.config['topk']
        if self.config['head_interaction'] in ['multiplicative', 'hierarchical']:
            self.medusa_num_heads = self.config['num_segment_head'] * self.config['num_prior_head']
        elif self.config['head_interaction'] == 'additive':
            self.medusa_num_heads = self.config['num_segment_head'] + self.config['num_prior_head']
        else:
            raise ValueError(f'Unknown head_interaction: {self.config["head_interaction"]}')
        self.split_mode = self.config['split_mode']
        self.device = self.config['device']
        self.all_tags = None
        self.eval_each_head = self.config.get('eval_each_head', False)
        if self.eval_each_head:
            assert self.config.get('all_tags', False)

    def set_all_tags(self, item_tags):
        self.all_tags = item_tags

    def reset_all_tags(self):
        del self.all_tags
        self.all_tags = None

    def data_collect(self, train_data):
        """ Collect the evaluation resource from training data.
            Args:
                train_data (AbstractDataLoader): the training dataloader which contains the training data.

        """
        if self.register.need('data.num_items'):
            for pred_idx in self.metrics_pred_len_list:
                self.data_struct[pred_idx].set('data.num_items', train_data.dataset.item_num)
        if self.register.need('data.num_users'):
            for pred_idx in self.metrics_pred_len_list:
                self.data_struct[pred_idx].set('data.num_users', train_data.dataset.user_num)
        if self.register.need('data.count_items'):
            for pred_idx in self.metrics_pred_len_list:
                self.data_struct[pred_idx].set('data.count_items', train_data.dataset.item_counter)
        if self.register.need('data.count_users'):
            for pred_idx in self.metrics_pred_len_list:
                self.data_struct[pred_idx].set('data.count_items', train_data.dataset.user_counter)

    def _average_rank(self, scores):
        """Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.

        Args:
            scores(tensor): an ordered tensor, with size of `(N, )`

        Returns:
            torch.Tensor: average_rank

        Example:
            >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]]))
            tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
            [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])

        Reference:
            https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352

        """
        length, width = scores.shape
        true_tensor = torch.full((length, 1), True, dtype=torch.bool, device=self.device)

        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        # bias added to dense
        bias = torch.arange(0, length, device=self.device).repeat(width).reshape(width, -1). \
            transpose(1, 0).reshape(-1)
        dense = obs.view(-1).cumsum(0) + bias

        # cumulative counts of each unique value
        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        # get average rank
        avg_rank = .5 * (count[dense] + count[dense - 1] + 1).view(length, -1)

        return avg_rank

    def eval_batch_collect(
        self, scores_tensor: torch.Tensor, positive_u: torch.Tensor, positive_i: torch.Tensor,
            tag_category: torch.Tensor=None, outlier_users: torch.Tensor=None, log_detailed_results=False
    ):
        """ Collect the evaluation resource from batched eval data and batched model output.
            Args:
                scores_tensor (Torch.Tensor): the output tensor of model with the shape of `(N, )`
                positive_u(Torch.Tensor): the row index of positive items for each user.
                positive_i(Torch.Tensor): the positive item id for each user.
                tag_category(Torch.Tensor): the tag category for each target_item.

            Collects for visualization:
                top_scores_by_head (dictionary)
                'values': top values from all heads
                'head_source': which head the top value comes from
                'idx': which items received the top values
                'values_by_head': top values by each head
                'idx_by_head': idx corresponding to 'values_by_head'
        """
        
        # Ensure all tensors are on the same device as scores_tensor
        B = scores_tensor.shape[0]

        scores_tensor = scores_tensor.float()

        if tag_category is not None:
            for pred_idx in self.metrics_pred_len_list:
                # tag_category: [user_batch, pred_len, num_categories]
                self.data_struct[pred_idx].update_tensor('rec.tgt_tags',
                                                         torch.any(tag_category[:, :pred_idx+1], dim=1))
            del tag_category

        if outlier_users is not None:
            self.data_struct[self.eval_pred_len-1].update_tensor('rec.outlier_users', outlier_users)
            del outlier_users

        if self.register.need('rec.items'):
            # get topk
            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)  # n_users x k
            for pred_idx in self.metrics_pred_len_list:
                self.data_struct[pred_idx].update_tensor('rec.items', topk_idx)

        if self.register.need('rec.topk'):
            self.count += 1
            # scores_tensor: [user_batch, num_heads, all_candidate_items]
            if self.count < 10:
                torch.save(scores_tensor, f'scores_tensor_merrec_{self.count}.pt')
            top_scores_by_head = dict()
            log_topk = 200
            top_k = max(self.topk)  # self.topk: [5, 10, 50, 200]
            if scores_tensor.shape[1] == 1:
                scores_tensor = scores_tensor.squeeze()
                topk_values, topk_idx = torch.topk(scores_tensor, top_k, dim=-1)  # n_users x k
                if log_detailed_results:
                    top_scores_by_head['values'] = topk_values[:, :log_topk].float().data.cpu().numpy()
                    top_scores_by_head['head_source'] = np.zeros((scores_tensor.shape[0], log_topk), dtype=int)
                    top_scores_by_head['idx'] = topk_idx[:, :log_topk].data.cpu().tolist()
                    
                    primary_num_cats = self.all_tags.shape[1]
                    idx_by_category = []
                    values_by_category = []
                    
                    for cat_idx in range(primary_num_cats):
                        category_mask = self.all_tags[:, cat_idx].bool()  # bool tensor
                        masked_scores = scores_tensor.clone()
                        masked_scores[:, ~category_mask] = float('-inf')  # mask False items to -inf
                        category_topk_values, category_topk_idx = torch.topk(masked_scores, log_topk, dim=-1)
                        idx_by_category.append(category_topk_idx)
                        values_by_category.append(category_topk_values)
                    
                    # Stack to create final tensors: [num_users, primary_num_cats, log_topk]
                    top_scores_by_head['idx_by_head'] = torch.stack(idx_by_category, dim=1).long().data.cpu().tolist()
                    top_scores_by_head['values_by_head'] = torch.stack(values_by_category, dim=1).float().data.cpu().numpy()
                    
            elif self.split_mode == 'average':
                finite_mask = torch.isfinite(scores_tensor)
                scores_tensor = torch.sum(scores_tensor.masked_fill(~finite_mask, 0), dim=1) / (torch.sum(finite_mask, dim=1) + 1e-8)
                topk_values, topk_idx = torch.topk(scores_tensor, top_k, dim=-1)  # n_users x k
                if log_detailed_results:
                    top_scores_by_head['values'] = topk_values[:, :log_topk].float().data.cpu().numpy()
                    top_scores_by_head['head_source'] = np.zeros((scores_tensor.shape[0], log_topk), dtype=int)
                    top_scores_by_head['idx'] = topk_idx[:, :log_topk].data.cpu().tolist()
                if self.eval_each_head:
                    topk_idx_head_list = []
                    for head_idx in range(self.medusa_num_heads):
                        topk_idx_head_list.append(torch.topk(scores_tensor[:, head_idx], k=top_k, dim=-1)[1])
                    topk_idx_head = torch.stack(topk_idx_head_list, dim=1)

            elif self.split_mode == 'combine':
                batch_size, medusa_num_heads, num_items = scores_tensor.shape

                # Step 1: Retrieve top-k items with a buffer for duplicates
                topk_values_per_head, topk_indices_per_head = torch.topk(scores_tensor, k=top_k, dim=-1)
                topk_head_source = torch.arange(medusa_num_heads, device=scores_tensor.device).view(1, medusa_num_heads, 1)
                topk_head_source = topk_head_source.expand(batch_size, medusa_num_heads, top_k)

                # Step 2: Flatten across heads for deduplication preparation
                # both have shape: (batch_size, medusa_num_heads * top_k)
                flattened_values = topk_values_per_head.view(batch_size, -1)
                flattened_indices = topk_indices_per_head.view(batch_size, -1)
                flattened_source = topk_head_source.reshape(batch_size, -1)

                # Step 3: Sort by score to prioritize high scores across the entire batch
                sorted_values, sorted_idx = flattened_values.sort(dim=-1, descending=True)
                sorted_indices = flattened_indices.gather(dim=-1, index=sorted_idx)
                sorted_source = flattened_source.gather(dim=-1, index=sorted_idx)

                # Step 4: Deduplicate while preserving the order of high scores
                # Create a mask for the first occurrence of each unique item across each batch
                seen_items = torch.zeros(batch_size, num_items, dtype=torch.bool, device=scores_tensor.device)
                is_unique = torch.zeros_like(sorted_indices, dtype=torch.bool)

                for i in range(sorted_indices.shape[1]):
                    current_indices = sorted_indices[:, i]
                    is_unique[:, i] = ~seen_items.gather(1, current_indices.unsqueeze(-1)).squeeze(-1)
                    seen_items.scatter_(1, current_indices.unsqueeze(-1), True)  # Mark the item as seen

                # Step 5: Apply the mask and get unique items in a batch-wise manner
                unique_values = [sorted_values[i][is_unique[i]][:top_k] for i in range(batch_size)]
                unique_items = [sorted_indices[i][is_unique[i]][:top_k] for i in range(batch_size)]
                unique_sources = [sorted_source[i][is_unique[i]][:top_k] for i in range(batch_size)]

                topk_idx = torch.stack(unique_items).long()
                # `topk_idx` now contains exactly `top_k` unique indices per sample, with padding as necessary
                if log_detailed_results:
                    top_scores_by_head['values'] = torch.stack(unique_values)[:, :log_topk].float().data.cpu().numpy()
                    top_scores_by_head['head_source'] = torch.stack(unique_sources)[:, :log_topk].data.cpu().numpy()
                    top_scores_by_head['idx'] = topk_idx[:, :log_topk].data.cpu().tolist()
                    top_scores_by_head['values_by_head'] = topk_values_per_head.float().data.cpu().numpy()
                    top_scores_by_head['idx_by_head'] = topk_indices_per_head.data.cpu().tolist()

            else:
                raise ValueError(f'Unknown split_mode: {self.split_mode}')

            del scores_tensor
            torch.cuda.empty_cache()

            # Check uniqueness for each row in a vectorized manner
            # Count unique elements in each row and compare to top_k
            unique_counts = torch.tensor([torch.unique(row).numel() for row in topk_idx], device=topk_idx.device)
            assert torch.all(unique_counts == top_k), "Duplicated elements found in some batch samples"

            if self.all_tags is not None:
                rec_tags = self.all_tags[topk_idx]
                self.data_struct[-1].update_tensor('rec.rec_tags', rec_tags)
                del rec_tags

            # count UNIQUE positives per user (Vectorized: sort then count changes)
            sorted_full, _ = positive_i.sort(dim=1)                  # (B, L)
            first_occ = torch.ones_like(sorted_full, dtype=torch.bool)
            first_occ[:, 1:] = sorted_full[:, 1:] != sorted_full[:, :-1]
            pos_len_full = first_occ.cumsum(dim=1).int()             # (B, L)
            del sorted_full, first_occ

            hit_mask = torch.zeros(B, top_k, dtype=torch.bool, device=topk_idx.device)
            prev_idx = 0
            for pred_idx in self.metrics_pred_len_list:
                # Build a slim boolean mask — still entirely on-GPU; Shape: (B, topk, pred_len)
                positive_slice = positive_i[:, prev_idx:pred_idx + 1]  # (B, L)
                hit_mask |= topk_idx.unsqueeze(-1).eq(positive_slice.unsqueeze(1)).any(dim=-1, keepdim=False)  # (num_users, topk, pred_len)
                pos_idx  = hit_mask.to(torch.uint8)  # (num_users, topk)

                result = torch.cat((pos_idx, pos_len_full[:, pred_idx:pred_idx+1]), dim=1)                      # (B, K+1)
                self.data_struct[pred_idx].update_tensor('rec.topk', result)

            if self.eval_each_head:
                for head_idx in range(self.medusa_num_heads):
                    pos_mask_head = topk_idx_head[:, head_idx].unsqueeze(-1).eq(positive_i.unsqueeze(1))  # (num_users, topk, pred_len)
                    pos_idx_head  = pos_mask_head.any(dim=-1).int()  # (num_users, topk)
                    result_head = torch.cat((pos_idx_head, pos_len_full[:, -1:]), dim=1)
                    self.data_struct[pred_idx].update_tensor(f'rec.topk_{head_idx}', result_head)

            return top_scores_by_head

        if self.register.need('rec.meanrank'):
            print('need mean rank')

            desc_scores, desc_index = torch.sort(scores_tensor, dim=-1, descending=True)
            avg_rank = self._average_rank(desc_scores)
            user_len_list = desc_scores.argmin(dim=1, keepdim=True)

            # get the index of positive items in the ranking list
            for pred_idx in self.metrics_pred_len_list:
                pos_matrix = torch.zeros_like(scores_tensor)
                for current_pred_idx in range(pred_idx + 1):  # use the next K items as target
                    pos_matrix[positive_u[:, current_pred_idx], positive_i[:, current_pred_idx]] = 1
                pos_index = torch.gather(pos_matrix, dim=1, index=desc_index)

                pos_rank_sum = torch.where(pos_index == 1, avg_rank, torch.zeros_like(avg_rank)).sum(dim=-1, keepdim=True)
                pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
                result = torch.cat((pos_rank_sum, user_len_list, pos_len_list), dim=1)
                self.data_struct[pred_idx].update_tensor('rec.meanrank', result)

        if self.register.need('rec.score'):
            print('need score')
            for pred_idx in self.metrics_pred_len_list:
                self.data_struct[pred_idx].update_tensor('rec.score', scores_tensor)

        # if self.register.need('data.label'):
            # self.label_field = self.config['LABEL_FIELD']
            # self.data_struct.update_tensor('data.label', interaction[self.label_field].to(self.device))

        return None

    def model_collect(self, model: torch.nn.Module):
        """ Collect the evaluation resource from model.
            Args:
                model (nn.Module): the trained recommendation model.
        """
        pass
        # TODO:

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        """ Collect the evaluation resource from total output and label.
            It was designed for those models that can not predict with batch.
            Args:
                eval_pred (torch.Tensor): the output score tensor of model.
                data_label (torch.Tensor): the label tensor.
        """
        if self.register.need('rec.score'):
            for pred_idx in self.metrics_pred_len_list:
                self.data_struct[pred_idx].update_tensor('rec.score', eval_pred)

        if self.register.need('data.label'):
            self.label_field = self.config['LABEL_FIELD']
            for pred_idx in self.metrics_pred_len_list:
                self.data_struct[pred_idx].update_tensor('data.label', data_label.to(self.device))

    def get_data_struct(self, pred_idx=0):
        """ Get all the evaluation resource that been collected.
            And reset some of outdated resource.
        """
        self.data_struct[pred_idx].finalize_tensors()
        returned_struct = copy.deepcopy(self.data_struct[pred_idx])
        key_list = ['rec.rec_tags', 'rec.tgt_tags', 'rec.outlier_users', 'rec.topk', 'rec.meanrank',
                    'rec.score', 'rec.items', 'data.label']
        if self.eval_each_head:
            for head_idx in range(self.medusa_num_heads):
                key_list.append(f'rec.topk_{head_idx}')
        for key in key_list:
            if key in self.data_struct[pred_idx]:
                del self.data_struct[pred_idx][key]
        return returned_struct