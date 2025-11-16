import copy
import gc
import os
from collections import Counter
from logging import getLogger
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch_geometric.utils import degree
from sklearn.preprocessing import MultiLabelBinarizer

from REC.utils import set_color
from REC.data.shareables import SharedList
import REC.data.comm as comm


class InteractionData:
    def __init__(self, config):
        self.config = config
        self.pred_len = config['pred_len']
        self.eval_pred_len = config['eval_pred_len']
        self.max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH'] + 1
        self.use_image_online = config['use_image_online']
        self.dataset_name = config['dataset']
        self.logger = getLogger()
        self.timestamp_required = config['timestamp_required']
        self.sample_last_only = config.get('sample_last_only', False)
        self.log_detailed_results = config.get('log_detailed_results', False)
        self.use_image = config['use_image']
        self.use_image_online = config['use_image_online']
        if self.use_image:
            self.image_dir = os.path.join(config['image_dir'], config['dataset'])
        else:
            self.image_dir = None
        self.category_by = config['category_by']
        self.eval_num_cats = config['eval_num_cats']
        self.train_test_gap = int(config.get('train_test_gap', 0))
        self.subset_user = config.get('subset_user', False)
        self.subset_user_div = config.get('subset_user_div', 10)
        self.subset_user_rmd = config.get('subset_user_rmd', 0)  # only keep uid % subset_user_div == subset_user_rmd
        self.cluster_as_tag = config.get("cluster_as_tag", False)
        if self.eval_num_cats > 1 and self.category_by == 'item':
            self.tag_col = f'cluster_{self.config["tag_version"]}' if self.cluster_as_tag else 'tag'
        elif self.eval_num_cats > 1 and self.category_by == 'user':
            assert self.cluster_as_tag, "cluster_as_tag must be True for user category"
            self.tag_col = f'user_cluster_{self.config["tag_version"]}'
        else:
            self.tag_col = None

        self.local_rank = comm.get_local_rank()
        self.uid_field = 'user_id'
        self.iid_field = 'item_id'
        self.user_seq: List[np.ndarray] = []
        self.time_seq: List[np.ndarray] = []
        self.event_seq: List[np.ndarray] = []
        self.valid_sample_locations = []
        self.train_seq_len = []  # uid --> length of training sequence
        self.train_feat = None
        self.id2token: Dict[str, List[str]] = {k: [] for k in ['user_id', 'item_id']}
        self.user_cluster_list: List[int] = [0]
        if self.eval_num_cats > 1:
            self.int_category_to_item_id: List[List[int]] = []
        self.item_to_info: List[Dict[str, str]] = [{}]  # placeholder for id 0
        self.item_interact_weights: List[int] = [0]
        self.item_weights_by_cat: List[List[int]] = [[0]]
        self.user_num, self.item_num, self.interact_num = 0, 0, 0
        self.counter = {}
        # defer all the actual loading to build()
        self.category_counts = dict()
        self.tag_to_category = dict()
        self.category_to_int = dict()

    def _from_scratch(self):
        self.logger.info(set_color(f'Loading {self.__class__} from scratch.', 'green'))
        interact_df = self._load_interact_feat(self.dataset_name)
        self._split_by_user(interact_df)
        self.logger.info(f'first ten user tokens: {self.id2token["user_id"][:10]}, first ten item tokens: {self.id2token["item_id"][:10]}')
        self.logger.info(f'user_seq: {self.user_seq[:10]}')
        del interact_df
        self._get_valid_sample_loc_for_train()  # valid sampling locations during training
        self._load_item_feat()

    def _load_interact_feat(self, token):
        interact_feat_path = os.path.join(self.config['data_path'], f'{token}.parquet')
        if not os.path.isfile(interact_feat_path):
            raise ValueError(f'File {interact_feat_path} not exist.')
        dtype_list = {'item_id': str, 'user_id': str, 'timestamp': int}
        name_list = ['item_id', 'user_id', 'timestamp']
        if self.category_by == 'event' and self.eval_num_cats > 1:
            self.logger.info('Category as defined per event. "event_id" column fetched...')
            dtype_list['event_id'] = int
            name_list.append('event_id')
        if self.category_by == 'user' and self.eval_num_cats > 1:
            self.logger.info(f'Category as defined per user. {self.tag_col} column fetched...')
            dtype_list[self.tag_col] = int
            name_list.append(self.tag_col)

        interact_df = pl.read_parquet(
            interact_feat_path, columns=name_list
        )
        self.logger.info(f'Interaction feature loaded successfully from [{interact_feat_path}].')
        self.logger.info(f'Interaction feature preview: {interact_df.head()}')

        # remove all users with fewer than min_seq_len or eval_pred_len * 2 interactions
        filter_min_len = self.eval_pred_len * 2
        if self.config['min_seq_len'] is not None:
            filter_min_len = max(self.config['min_seq_len'], filter_min_len)
        interact_df = interact_df.with_columns(pl.col("item_id").list.len().alias("num_interacts"))
        interact_df = interact_df.filter(pl.col("num_interacts") > filter_min_len)
        interact_df = interact_df.drop("num_interacts")
        if self.log_detailed_results:
            interact_feat_path = os.path.join(self.config['data_path'], f'{token}-sub256.parquet')
            interact_df_sub = pl.read_parquet(
                interact_feat_path, columns=name_list
            )
        else:
            interact_df_sub = None
        interact_df = self._build_token_id_maps(interact_df, interact_df_sub)
        if self.category_by == 'user' and self.eval_num_cats > 1:
            self.user_cluster_list = [0] + pl.Series(
                interact_df.select(pl.col(self.tag_col))).to_list()

        self.user_num = len(self.id2token[self.uid_field])
        self.item_num = len(self.id2token[self.iid_field])
        self.logger.info(f'len interact df = {len(interact_df)}')
        self.logger.info(f"{self.user_num = } {self.item_num = }")

        self.interact_num = interact_df.select(pl.col(self.iid_field).list.len().sum()).item()
        return interact_df

    def _build_token_id_maps(self, interact_df, interact_df_sub):
        if interact_df_sub is not None:
            self.id2token[self.uid_field] = ['[PAD]'] + pl.Series(
                interact_df_sub.select(pl.col(self.uid_field))).to_list()
        else:
            self.id2token[self.uid_field] = ['[PAD]'] + pl.Series(
                interact_df.select(pl.col(self.uid_field))).to_list()
        self.id2token[self.iid_field] = ['[PAD]'] + sorted(list(set(pl.Series(
            interact_df.select(pl.col(self.iid_field).list.explode())).to_list())))
        token_id = {t: i + 1 for i, t in enumerate(self.id2token[self.uid_field][1:])}
        if interact_df_sub is not None:
            interact_df_sub = interact_df_sub.with_columns(
                pl.col(self.uid_field).replace_strict(token_id, default=-1).alias(self.uid_field))
            # we will map item_id in def _split_by_user
            return interact_df_sub
        else:
            interact_df = interact_df.with_columns(
                pl.col(self.uid_field).replace_strict(token_id, default=-1).alias(self.uid_field))
            return interact_df

    def _split_by_user(self, interact_df):
        self.user_seq = [[]] + pl.Series(interact_df.select(self.iid_field)).to_list()
        token_id = {t: i for i, t in enumerate(self.id2token[self.iid_field])}
        _get = token_id.__getitem__
        self.user_seq = [list(map(_get, sublist)) for sublist in self.user_seq]
        if self.timestamp_required:
            self.time_seq = [[]] + pl.Series(interact_df.select('timestamp')).to_list()
        if self.category_by == 'event' and self.eval_num_cats > 1:
            self.event_seq = [[]] + pl.Series(interact_df.select('event_id')).to_list()

    def _get_valid_sample_loc_for_train(self):
        for uid in range(self.user_num):
            self.train_seq_len.append(len(self.user_seq[uid]) - self.config['eval_pred_len'] * 2 - self.train_test_gap)
            if self.subset_user and uid % 10 != self.subset_user_rmd:
                continue
            if self.train_seq_len[uid] <= 1:
                if uid < 30:
                    self.logger.info(f'ID = {self.id2token["user_id"][uid]} has no valid samples. len = {len(self.user_seq[uid])}')
                continue
            if self.sample_last_only:  # we only sample one window per user for Amazon Books
                if self.train_seq_len[uid] < self.pred_len + 3:  # 3 is an arbitrary minimum context length
                    self.valid_sample_locations.append((uid, self.train_seq_len[uid] - 1))
                else:
                    self.valid_sample_locations.append((uid, self.train_seq_len[uid] - self.pred_len))
                if uid < 30:
                    self.logger.info(f'ID = {self.id2token["user_id"][uid]} has sample location={self.valid_sample_locations[-1]}. len = {len(self.user_seq[uid])}')
            elif self.train_seq_len[uid] <= self.max_item_list_len:  # Pixel
                # -1 since we need at least one item in pred
                if uid < 30:
                    self.logger.info(f'ID = {self.id2token["user_id"][uid]} has sample location={(uid, self.train_seq_len[uid] - 1)}. len = {len(self.user_seq[uid])}')
                self.valid_sample_locations.append((uid, self.train_seq_len[uid] - 1))
            else:
                # non-overlap
                offset = (self.train_seq_len[uid] - 1) % self.max_item_list_len
                if uid < 30:
                    new_locations = [(uid, context_end) for context_end in
                                     range(offset, self.train_seq_len[uid], self.max_item_list_len)]
                    self.logger.info(
                        f'ID = {self.id2token["user_id"][uid]} has sample location={new_locations}. len = {len(self.user_seq[uid])}')
                self.valid_sample_locations += [(uid, context_end) for context_end in
                                                range(offset, self.train_seq_len[uid], self.max_item_list_len)]
        self.logger.info(f'valid_sample_locations: {len(self.valid_sample_locations)}')

    def _load_item_feat(self):
        if not self.config['text_path'].endswith('.parquet'):
            raise ValueError(f"Unsupported file format for item features: {self.config['text_path']}")

        df = pd.read_parquet(self.config['text_path'])
        item_keys = self.config['text_keys'] + ['item_id']
        if self.tag_col is not None and self.category_by in ['item'] and self.tag_col not in item_keys:
            item_keys += [self.tag_col]
        if self.config['use_image'] and self.config['use_image_online']:
            item_keys += ['image']
        if self.config.get('neg_sample_mode', None) is not None:
            item_keys += ['interact_count']
        df = df[item_keys]
        set_of_items = set(self.id2token['item_id'])
        df = df[df['item_id'].isin(set_of_items)].reset_index(drop=True)

        if self.use_image:
            if not self.use_image_online:
                df['image'] = self.image_dir.rstrip('/') + '/' + df['item_id'].astype(str) + '.jpg'
                exists = df['image'].map(os.path.exists)
                df.loc[~exists, 'image'] = None
                print(f'df[image]: {df["image"]}')
            else:
                # assume df['image'] already holds URLs or other values; null out non-URLs
                df['image'] = df['image'].where(
                    df['image'].astype(str).str.startswith(('http://', 'https://')), None)
        else:
            df['image'] = None

        if self.eval_num_cats > 1 and self.category_by == 'item':
            # map tag to category
            ordered_category = [self.config['int_to_category'][i] for i in range(self.config['eval_num_cats'])]
            self.logger.info(f'ordered_category: {ordered_category}')

            if self.cluster_as_tag:
                df['category_list'] = df[self.tag_col].apply(lambda t: self.tag_to_category.get(t, []))
            else:
                df['category_list'] = df[self.tag_col].apply(lambda t: self.tag_to_category.get(t, []))
            mlb = MultiLabelBinarizer(classes=ordered_category)
            one_hot = mlb.fit_transform(df['category_list'])
            df['original_tag_category'] = one_hot.astype(bool).tolist()
            df = df.drop(columns='category_list')
            if self.config.get("random_tags", False):
                self.logger.info('*** Ablation experiment: randomly assigning items to categories ***')
                rng = np.random.default_rng(seed=42)  # seed for reproducibility
                random_cats = rng.integers(0, 2, size=(len(df), len(ordered_category)), dtype=np.int64)
                df['tag_category'] = [row.tolist() for row in random_cats]
            elif self.config.get("all_tags", False):
                self.logger.info('*** Ablation experiment: assigning each item to all categories ***')
                all_true_cats = np.ones((len(df), len(ordered_category)), dtype=bool).tolist()
                df['tag_category'] = [row for row in all_true_cats]
            else:
                df['tag_category'] = df['original_tag_category'].apply(copy.deepcopy)
            no_cats = df.loc[df['tag_category'].apply(sum) == 0, self.tag_col].unique()
            if len(no_cats):
                self.logger.warning(f"These tags had no mapping: {no_cats.tolist()}")

        df.rename(columns={'item_id': 'old_item_id'}, inplace=True)
        self.logger.info(f'id2token first ten: {self.id2token["item_id"][:10]}')
        reverse_id_map = {t: i for i, t in enumerate(self.id2token['item_id'])}
        self.logger.info(f'id2token reverse: {[reverse_id_map[idx] for idx in self.id2token["item_id"][:10]]}')
        df['item_id'] = df['old_item_id'].map({t: i for i, t in enumerate(self.id2token['item_id'])})
        df = df.drop(columns='old_item_id')
        records = df.to_dict(orient='records')
        by_id = {rec['item_id']: rec for rec in records}
        self.item_to_info.extend([by_id.get(i, {}) for i in range(1, self.item_num)])
        if self.config['neg_sample_mode'] is not None:
            self.logger.info(f'WARNING: neg_sample_mode={self.config["neg_sample_mode"]} is not None. This can slow down your code!')
            item_interact_count = np.array([by_id.get(i, {}).get('interact_count', 0)
                                            for i in range(1, self.item_num)])
            if self.config['neg_sample_mode'] == 'identity':
                item_interact_weights = item_interact_count
            elif self.config['neg_sample_mode'] == 'sqrt':
                item_interact_weights = np.sqrt(item_interact_count)
            elif self.config['neg_sample_mode'] == 'log':
                item_interact_weights = np.log(item_interact_count + 1)
            else:
                raise ValueError(f"Unsupported neg_sample_mode: {self.config['neg_sample_mode']}")
            del item_interact_count
            item_interact_weights = np.cumsum(item_interact_weights)
            self.item_interact_weights = (item_interact_weights / item_interact_weights[-1]).tolist()
            print('item_interact_weights: ', self.item_interact_weights[-10:])
            del item_interact_weights
        for i in range(self.item_num):
            if i not in by_id:
                self.logger.info(f'WARNING!!!! item_id={i} has no info')
        for i in range(10):
            self.logger.info(f'item={i} info: {self.item_to_info[i]}')
        del set_of_items, records, by_id

        if self.config['eval_num_cats'] > 1 and self.category_by == 'item':
            """
            To build self.int_category_to_item_id, which is useful when drawing negative samples from the same category
            """
            assert self.tag_col in item_keys
            if self.config['neg_sample_mode'] is not None:
                tag_df = df[['item_id', self.tag_col, 'interact_count']].copy()
            else:
                tag_df = df[['item_id', self.tag_col]].copy()
            tag_df['categories'] = tag_df[self.tag_col].map(self.tag_to_category)

            # 3) explode so each row has exactly one category
            df_expanded = tag_df.explode('categories').dropna(subset=['categories'])

            # 4) group by category and collect item_ids into a list
            num_classes = len(self.config['int_to_category'])
            self.logger.info(f'num_classes: {num_classes}')
            expected = set(range(num_classes))
            actual = set(self.config['int_to_category'].keys())
            if self.config['neg_sample_mode'] is not None:
                grouped = df_expanded.groupby('categories', sort=False)[['item_id', 'interact_count']].agg(list)
                category_to_item_id = grouped['item_id'].to_dict()
                category_to_interact_count = grouped['interact_count'].to_dict()
                item_interact_count: List[List[int]] = [
                     category_to_interact_count.get(self.config['int_to_category'][i], []) for i in range(num_classes)
                ]
                if self.config['neg_sample_mode'] == 'identity':
                    self.item_weights_by_cat = [np.array(item_interact_count[i]) for i in range(num_classes)]
                elif self.config['neg_sample_mode'] == 'sqrt':
                    self.item_weights_by_cat = [np.sqrt(item_interact_count[i]) for i in range(num_classes)]
                elif self.config['neg_sample_mode'] == 'log':
                    self.item_weights_by_cat = [np.log(np.array(item_interact_count[i]) + 1)
                                                for i in range(num_classes)]
                else:
                    raise ValueError(f"Unsupported neg_sample_mode: {self.config['neg_sample_mode']}")
                del item_interact_count
                for i in range(num_classes):
                    self.item_weights_by_cat[i] = np.cumsum(self.item_weights_by_cat[i])
                    self.item_weights_by_cat[i] = self.item_weights_by_cat[i] / self.item_weights_by_cat[i][-1]
                    self.item_weights_by_cat[i] = self.item_weights_by_cat[i].tolist()
                    print('item_weights_by_cat', self.item_weights_by_cat[i][-10:])
            else:
                category_to_item_id = df_expanded.groupby('categories')['item_id'].apply(list).to_dict()
            assert actual == expected, (
                f"config[int_to_category] keys must be 0..{num_classes - 1}, but got {sorted(actual)}"
            )
            self.int_category_to_item_id: List[List[int]] = [
                category_to_item_id.get(self.config['int_to_category'][i], []) for i in range(num_classes)
            ]
            for cat_idx in range(self.config['eval_num_cats']):
                for uid in range(10):
                    item_id = self.int_category_to_item_id[cat_idx][uid]
                    self.logger.info(f'cat={self.config["int_to_category"][cat_idx]} uid={uid}, item={item_id}, item_info={self.item_to_info[item_id]}')
            del tag_df, df_expanded, category_to_item_id
        else:
            self.int_category_to_item_id = None
        del df
        self.logger.info(f"Loaded item features from {self.config['text_path']}. "
                         f"Text Item num: {len(self.item_to_info)}")

    def build(self):
        # This part takes little memory, and thus runs on all ranks
        if self.config['eval_num_cats'] > 1:
            import importlib
            if self.cluster_as_tag:
                if self.category_by == 'user':
                    module_name = f"REC.data.{self.dataset_name}_user_cluster_dict"
                else:
                    module_name = f"REC.data.{self.dataset_name}_cluster_dict"
            else:
                module_name = f"REC.data.{self.dataset_name}_tag_dict"
            tag_mod = importlib.import_module(module_name)
            tag_to_general = tag_mod.tag_to_general
            if self.category_by in ['item', 'user']:
                self.category_counts = tag_to_general[self.config['tag_version']]['category_counts']
                self.tag_to_category = tag_to_general[self.config['tag_version']]['tag_to_category']
                unique_categories = sorted(self.category_counts.keys())
                self.category_to_int = {cat: idx for idx, cat in enumerate(unique_categories)}
                self.config['int_to_category'] = {value: key for key, value in self.category_to_int.items()}
            elif self.category_by == 'event':
                self.category_counts = tag_to_general['category_counts']
                self.category_to_int = tag_to_general['category_to_int']
                self.config['int_to_category'] = {value: key for key, value in self.category_to_int.items()}
            else:
                raise ValueError(f'category_by = {self.category_by} is not defined.')

        # only perform the actual operation in local rank 0 and broadcast to other ranks
        if self.local_rank == 0:
            self.logger.info(f"build {self.dataset_name} dataload on rank {self.local_rank}")
            self._from_scratch()
        else:
            # load from local rank 0
            self.logger.info(f"skip load {self.dataset_name} dataload on rank {self.local_rank}")
        self.logger.info(f"Broadcast user_seq in dataload")
        self.user_seq = SharedList(self.user_seq)
        if self.timestamp_required:
            self.logger.info(f"Broadcast time_seq in dataload")
            self.time_seq = SharedList(self.time_seq)
        self.logger.info(f"Broadcast valid_sample_locations in dataload")
        self.valid_sample_locations  = SharedList(self.valid_sample_locations)
        self.logger.info(f"Broadcast train_seq_len in dataload")
        self.train_seq_len  = SharedList(self.train_seq_len)
        self.logger.info(f"Broadcast id2token in dataload")
        self.id2token = {k: SharedList(v) for k, v in self.id2token.items()}
        self.logger.info(f"Broadcast item_interact_weights in dataload")
        self.item_interact_weights = SharedList(self.item_interact_weights)
        self.logger.info(f"Broadcast item_weights_by_cat in dataload")
        self.item_weights_by_cat = SharedList(self.item_weights_by_cat)
        if self.config['eval_num_cats'] > 1:
            if self.category_by == 'item':
                self.logger.info(f"Broadcast int_category_to_item_id in dataload")
                self.int_category_to_item_id = SharedList(self.int_category_to_item_id)
            elif self.category_by == 'user':
                self.logger.info(f"Broadcast user_cluster_list in dataload")
                self.user_cluster_list = SharedList(self.user_cluster_list)
            else:
                self.logger.info(f"Broadcast event_seq in dataload")
                self.event_seq = SharedList(self.event_seq)

        self.logger.info(f"Broadcast item_to_info dataload")
        self.item_to_info = SharedList(self.item_to_info)

        # some attributes were computed only on rank=0, so we recompute
        self.user_num = len(self.id2token[self.uid_field])
        self.item_num = len(self.id2token[self.iid_field])
        self.interact_num = sum(map(len, self.user_seq))

        self.counter = {
            "user_id": Counter({key: len(value) for key, value in enumerate(self.user_seq)}),
            "item_id": Counter(item for values in self.user_seq for item in values),
        }
        self.logger.info(f"{self.user_num = }, {self.item_num = }, {self.interact_num = }")
        self.logger.info(f"{len(self.user_seq) = }")
        gc.collect()

    @staticmethod
    def sort(inter_feat, by, ascending=True):
        if isinstance(inter_feat, pd.DataFrame):
            inter_feat.sort_values(by=by, ascending=ascending, inplace=True)

        else:
            if isinstance(by, str):
                by = [by]

            if isinstance(ascending, bool):
                ascending = [ascending]

            if len(by) != len(ascending):
                if len(ascending) == 1:
                    ascending = ascending * len(by)
                else:
                    raise ValueError(f'by [{by}] and ascending [{ascending}] should have same length.')
            for b, a in zip(by[::-1], ascending[::-1]):
                index = np.argsort(inter_feat[b], kind='stable')
                if not a:
                    index = index[::-1]
                for k in inter_feat:
                    inter_feat[k] = inter_feat[k][index]

    @property
    def avg_actions_of_users(self):
        """Get the average number of users' interaction records.

        Returns:
            numpy.float64: Average number of users' interaction records.
        """
        return self.interact_num / self.user_num

    @property
    def avg_actions_of_items(self):
        """Get the average number of items' interaction records.

        Returns:
            numpy.float64: Average number of items' interaction records.
        """
        return self.interact_num / self.item_num

    @property
    def sparsity(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.interact_num / self.user_num / self.item_num

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [set_color(self.dataset_name, 'pink')]
        if self.uid_field:
            info.extend([
                set_color('The number of users', 'blue') + f': {self.user_num}',
                set_color('Average actions of users', 'blue') + f': {self.avg_actions_of_users}'
            ])
        if self.iid_field:
            info.extend([
                set_color('The number of items', 'blue') + f': {self.item_num}',
                set_color('Average actions of items', 'blue') + f': {self.avg_actions_of_items}'
            ])
        info.append(set_color('The number of interactions', 'blue') + f': {self.interact_num}')
        if self.uid_field and self.iid_field:
            info.append(set_color('The sparsity of the dataset', 'blue') + f': {self.sparsity * 100}%')

        return '\n'.join(info)

    def copy(self, new_inter_feat):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.

        Args:
            new_inter_feat (Interaction): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    @property
    def user_counter(self):
        return self.counter['user_id']

    @property
    def item_counter(self):
        return self.counter['item_id']

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = torch.tensor(self.train_feat[self.uid_field])
        col = torch.tensor(self.train_feat[self.iid_field]) + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num + self.item_num)

        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight
