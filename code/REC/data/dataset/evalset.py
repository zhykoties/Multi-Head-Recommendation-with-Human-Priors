import torch
from torch.utils.data import Dataset
import datetime
import pytz
import torch.nn.functional as F


class SeqEvalDataset(Dataset):
    def __init__(self, config, dataload, phase='valid'):
        self.dataload = dataload
        self.user_num = self.dataload.user_num - 1  # minus the padded item
        self.eval_pred_len = config['eval_pred_len']
        self.timestamp_required = config['timestamp_required']
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH_TEST'] if config['MAX_ITEM_LIST_LENGTH_TEST'] else config['MAX_ITEM_LIST_LENGTH']
        self.phase = phase
        self.length = len(self.dataload.user_seq)
        self.item_num = dataload.item_num
        self.return_tag_mask = True if config["eval_num_cats"] > 1 else False
        self.category_by = config['category_by']
        assert self.category_by in ['item', 'event', 'user']
        if self.category_by == 'user':
            num_user_clusters = max(self.dataload.category_to_int.values()) + 1
            self.one_hot_user_cluster = F.one_hot(torch.tensor(self.dataload.user_cluster_list, dtype=torch.int64), num_user_clusters)
        self.outlier_user_metrics = config['outlier_user_metrics']
        if self.return_tag_mask:
            self.ordered_category = [config['int_to_category'][i] for i in range(config['eval_num_cats'])]
        else:
            self.ordered_category = []
        self.eval_num_cats = config['eval_num_cats']

    def __len__(self):
        return self.user_num

    def _padding_sequence(self, sequence, data_type):
        seq_len = sequence.shape[0]
        if seq_len >= self.max_item_list_length:
            return sequence[-self.max_item_list_length:]
        else:
            result = torch.zeros(self.max_item_list_length, dtype=data_type)
            result[self.max_item_list_length - seq_len:] = sequence
            return result

    def _padding_time_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0 for _ in range(pad_len)] + sequence
        sequence = sequence[-max_length:]
        vq_time = []
        for time in sequence:
            dt = datetime.datetime.fromtimestamp(time, pytz.timezone('UTC'))
            vq_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])
        return vq_time

    def process_item(self, item_id, fix_miscellaneous=False):
        item = self.dataload.item_to_info[item_id]
        if len(item) > 0:
            tag_category = item['tag_category']
            if fix_miscellaneous and all(tag_category):
                return [False for _ in self.ordered_category]
        else:
            tag_category = [False for _ in self.ordered_category]
        return tag_category

    def get_item_tag(self, item_id):
        item = self.dataload.item_to_info[item_id]
        return item['tag'] if len(item) > 0 else None

    def outlier_by_tag(self, history_seq, item_target):
        context_tags = set()
        for item in history_seq:
            tag = self.get_item_tag(item)
            if tag is not None:
                context_tags.add(tag)

        for item in item_target:
            tag = self.get_item_tag(item)
            if tag is not None and tag not in context_tags:
                return True

        return False

    def __getitem__(self, uid):
        uid = uid + 1
        if self.phase == 'valid':
            last_num = self.dataload.train_seq_len[uid]
            history_seq = torch.as_tensor(self.dataload.user_seq[uid][:last_num], dtype=torch.int64)
            item_target = self.dataload.user_seq[uid][last_num:last_num+self.eval_pred_len]
        else: 
            last_num = self.eval_pred_len
            history_seq = torch.as_tensor(self.dataload.user_seq[uid][:-last_num], dtype=torch.int64)
            item_target = self.dataload.user_seq[uid][-last_num:]
        item_seq = self._padding_sequence(history_seq, data_type=torch.int64)

        if self.timestamp_required:
            if self.phase == 'valid':
                history_time_seq = self.dataload.time_seq[uid][:last_num]
            else:
                history_time_seq = self.dataload.time_seq[uid][:-last_num]
            time_seq = self._padding_time_sequence(history_time_seq, self.max_item_list_length)
        else:
            time_seq = []

        target_categories = None
        any_uncovered = False
        if self.category_by == 'item':
            if self.outlier_user_metrics == 'category':
                context_categories = [self.process_item(int(item_id), fix_miscellaneous=True) for item_id in history_seq]
                context_coverage = [any(cat_flags) for cat_flags in zip(*context_categories)]
                target_categories_fixed = [self.process_item(int(item_id), fix_miscellaneous=True) for item_id in item_target]
                any_uncovered = any(
                    tc and not context_coverage[k] for row in target_categories_fixed for k, tc in enumerate(row)
                )
            elif self.outlier_user_metrics == 'tag':
                any_uncovered = self.outlier_by_tag(history_seq, item_target)
            if self.return_tag_mask:
                target_tag_categories = [self.process_item(int(item_id)) for item_id in item_target]
            else:
                target_tag_categories = []

        elif self.category_by == 'user':
            if self.return_tag_mask:
                target_tag_categories = self.one_hot_user_cluster[uid].unsqueeze(0).expand(self.eval_pred_len, -1).tolist()
            else:
                target_tag_categories = []

        else:  # category by event
            if self.outlier_user_metrics == 'event':
                if self.phase == 'valid':
                    history_events = self.dataload.event_seq[uid][:last_num]
                else:
                    history_events = self.dataload.event_seq[uid][:-last_num]
                if len(history_events) > self.max_item_list_length:
                    history_events = history_events[-self.max_item_list_length:]
                context_events = set(history_events)
                if self.phase == 'valid':
                    target_categories = self.dataload.event_seq[uid][last_num:last_num+self.eval_pred_len]
                else:
                    target_categories = self.dataload.event_seq[uid][-last_num:]
                for event in target_categories:
                    if event not in context_events:
                        any_uncovered = True
                        break

            if self.return_tag_mask:
                if target_categories is None:
                    if self.phase == 'valid':
                        target_categories = self.dataload.event_seq[uid][last_num:last_num+self.eval_pred_len]
                    else:
                        target_categories = self.dataload.event_seq[uid][-last_num:]
                target_tag_categories = [[(i == idx) for i in range(self.eval_num_cats)] for idx in target_categories]
            else:
                target_tag_categories = []

        # item_seq is the zero-padded version of history_seq
        return uid, history_seq, item_seq, torch.as_tensor(item_target, dtype=torch.int64), time_seq, target_tag_categories, \
            any_uncovered
