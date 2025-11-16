from asyncio.log import logger

from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import AutoProcessor
import base64
import random
import datetime
import pytz
import math
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import requests
from REC.data.qwen_vl_utils import process_vision_info
from REC.llm_dict import use_image_dict


class SEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.item_num = dataload.item_num
        self.config = config
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.pred_len = config['pred_len']
        self.window_len = self.pred_len + self.max_seq_length

        # multihead with prior settings
        self.return_tag_mask = True if config['loss'] == 'prior' else False
        self.category_by = config['category_by']
        self.neg_sample_mix_ratio = config['neg_sample_mix_ratio']
        self.neg_sample_by_cat = self.return_tag_mask and config['neg_sample_by_cat'] and self.category_by == 'item'
        self._all_items_arr = np.arange(1, self.item_num, dtype=np.int64)
        self.rng = np.random.default_rng()

        self.length = len(self.dataload.valid_sample_locations)
        self.item_list = dataload.id2token['item_id']
        self.eval_num_cats = config['eval_num_cats']
        self.use_neg_sample_weights = self.config['neg_sample_mode'] is not None
        logger.info(f'use_neg_sample_weights = {self.use_neg_sample_weights}')
        self.item_interact_weights = dataload.item_interact_weights
        self.item_weights_by_cat = dataload.item_weights_by_cat
        if self.category_by == 'user':
            print(f'num_users: {len(dataload.user_cluster_list)}, actual num_users: {len(dataload.id2token["user_id"])}')
            num_user_clusters = max(dataload.category_to_int.values()) + 1
            self.one_hot_user_cluster = F.one_hot(torch.tensor(dataload.user_cluster_list, dtype=torch.int64), num_user_clusters)
        if self.return_tag_mask and self.category_by == 'item':
            self._cat_item_arr = [np.asarray(lst, dtype=np.int64) for lst in dataload.int_category_to_item_id]
            for cat_idx in range(self.eval_num_cats):
                logger.info(f'Category{cat_idx}={config["int_to_category"][cat_idx]} has #items={len(self._cat_item_arr[cat_idx])}')
            logger.info(f'category_counts: {dataload.category_counts}')
            assert self.eval_num_cats == len(self._cat_item_arr)

        self.device = config['device']
        self.random_sample = config['pad_random_sample']
        self.num_negatives = config['num_negatives']
        if self.num_negatives:
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])
        else:
            if config['loss'] in ['nce', 'prior']:
                self.num_negatives = self.max_seq_length
        if not self.random_sample:
            logger.info(f"[Warning] --- Not using random samples for padding")

    def __len__(self):
        return self.length

    def _neg_sample(self, item_set, tag_category=None, k=1):
        # choose pool
        if tag_category is not None and self.rng.random() > self.neg_sample_mix_ratio:
            pool_arr = self._cat_item_arr[tag_category]
            weights = self.item_weights_by_cat[tag_category] if self.use_neg_sample_weights else None
        else:
            pool_arr = self._all_items_arr
            weights = self.item_interact_weights if self.use_neg_sample_weights else None

        if not isinstance(item_set, set):
            item_set = set(item_set)

        if not self.use_neg_sample_weights:
            draw_sz = min(pool_arr.size, k + len(item_set))
            draw = self.rng.choice(pool_arr, size=draw_sz, replace=False)
            if item_set:
                mask = ~np.isin(draw, list(item_set), assume_unique=False)
                out = draw[mask].tolist()
            else:
                out = draw.tolist()

            # Top-up at most once (rarely needed)
            need = k - len(out)
            if need > 0:
                extra = self.rng.choice(pool_arr, size=need, replace=False)
                # (collision with blacklist is extremely unlikely; if paranoid, filter again)
                out.extend(extra.tolist()[:need])
            return out[:k]


        # ---- Weighted path (kept simple; rarely used) ----
        res = []
        while len(res) < k:
            need = k - len(res)
            candidates = random.choices(pool_arr.tolist(), weights=weights, k=need)  # use weights=, not cum_weights=
            for x in candidates:
                if x not in item_set:
                    res.append(x)
        return res[:k]


    def _padding_sequence(self, item_seq, context_pad_len, pred_pad_len, data_type,
                          item_set=None, random_sample=False):
        # [pad, ..., pad, seq, ..., seq, pad, ..., pad]
        if random_sample:
            result = torch.empty(self.window_len, dtype=data_type)
            result[:context_pad_len] = torch.as_tensor(self._neg_sample(item_set, k=context_pad_len), dtype=data_type)
            result[context_pad_len:self.window_len - pred_pad_len] = item_seq
            result[self.window_len - pred_pad_len:] = torch.as_tensor(self._neg_sample(item_set, k=pred_pad_len), dtype=data_type)
        else:
            result = torch.zeros(self.window_len, dtype=data_type)
            result[context_pad_len:self.window_len-pred_pad_len] = item_seq
        return result

    def reconstruct_train_data(self, item_seq, context_pad_len, pred_pad_len):
        item_set = set(item_seq)
        item_seq = self._padding_sequence(torch.as_tensor(item_seq, dtype=torch.int64), context_pad_len, pred_pad_len,
                                          torch.int64, item_set=item_set, random_sample=self.random_sample)
        if self.neg_sample_by_cat:
            neg_item = [self._neg_sample(item_seq, cat_idx, k=self.num_negatives)
                        for cat_idx in range(self.eval_num_cats)]
            neg_item += [self._neg_sample(item_seq, k=self.num_negatives)]
            neg_item = torch.tensor(neg_item, dtype=torch.int64)
        else:
            neg_item = torch.tensor([self._neg_sample(item_seq, k=self.num_negatives)], dtype=torch.int64)
        masked_index = torch.zeros(self.window_len, dtype=torch.int64)
        masked_index[context_pad_len:self.window_len-pred_pad_len] = 1  # use 0's to indicate padded positions
        return item_seq, neg_item, masked_index

    def process_item(self, item_id):
        item = self.dataload.item_to_info[item_id]
        if len(item) > 0:
            tag_category = item['tag_category']
        else:
            tag_category = [False for _ in range(self.eval_num_cats)]
        return tag_category

    def process_event(self, event_seq, context_pad_len, pred_pad_len):
        event_mask = torch.zeros(self.window_len, self.eval_num_cats, dtype=torch.int64)
        event_categories = [[(i == idx) for i in range(self.eval_num_cats)]
                            for idx in event_seq[context_pad_len:self.window_len-pred_pad_len]]
        event_categories = torch.tensor(event_categories, dtype=torch.int64)
        event_mask[context_pad_len:self.window_len-pred_pad_len] = event_categories
        return event_mask

    def __getitem__(self, index):
        uid, context_end = self.dataload.valid_sample_locations[index]
        context_start = max(0, context_end - self.max_seq_length)
        context_pad_len = self.max_seq_length - context_end + context_start
        pred_len = min(self.dataload.train_seq_len[uid] - context_end, self.pred_len)
        item_seq = self.dataload.user_seq[uid][context_start:context_end + pred_len]  # python list
        item_seq, neg_item_seq, masked_index = self.reconstruct_train_data(item_seq, context_pad_len,
                                                                           self.pred_len - pred_len)

        if self.return_tag_mask:
            if self.category_by == 'item':
                tag_categories = torch.tensor([self.process_item(int(item_id)) for item_id in item_seq],
                                                     dtype=torch.int64)
            elif self.category_by == 'user':
                tag_categories = self.one_hot_user_cluster[uid].unsqueeze(0).expand(self.window_len, -1)
            else:
                event_seq = self.dataload.event_seq[uid][context_start:context_end + pred_len]
                event_seq = self._padding_sequence(torch.as_tensor(event_seq, dtype=torch.int64), context_pad_len,
                                                    self.pred_len - pred_len, torch.int64)
                tag_categories = self.process_event(event_seq, context_pad_len, self.pred_len - pred_len)
        else:
            tag_categories = torch.tensor([], dtype=torch.int64)
        return item_seq, neg_item_seq, masked_index, tag_categories


class TextSEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.item_num = self.dataload.item_num
        self.config = config
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.pred_len = config['pred_len']
        self.window_len = self.pred_len + self.max_seq_length
        self.img_height = config['img_height']
        self.img_width = config['img_width']
        self.timestamp_required = config['timestamp_required']
        self.freeze_item_llm = config.get('freeze_item_llm', False)

        # multihead with prior settings
        self.return_tag_mask = True if config['loss'] == 'prior' else False
        self.category_by = config['category_by']
        self.neg_sample_mix_ratio = config['neg_sample_mix_ratio']
        self.neg_sample_by_cat = self.return_tag_mask and config['neg_sample_by_cat'] and self.category_by == 'item'
        self._all_items_arr = np.arange(1, self.item_num, dtype=np.int64)
        self.rng = np.random.default_rng()

        idx = config['item_pretrain_dir'].rfind('/')
        if idx == -1:
            item_llm_name = config['item_pretrain_dir']
        else:
            item_llm_name = config['item_pretrain_dir'][idx + 1:]

        self.use_image = config['use_image']
        if self.use_image:
            can_use_img = use_image_dict[item_llm_name]['use_image']
            assert can_use_img, f"item_llm={item_llm_name} doesn't support images..."
        self.use_image_online = config['use_image_online']
        self.message_type = use_image_dict[item_llm_name]['message_type']
        self.has_chat_template = use_image_dict[item_llm_name]['has_chat_template']

        self.length = len(self.dataload.valid_sample_locations)
        self.item_list = dataload.id2token['item_id']
        self.eval_num_cats = config['eval_num_cats']
        self.use_neg_sample_weights = self.config['neg_sample_mode'] is not None
        logger.info(f'use_neg_sample_weights = {self.use_neg_sample_weights}')
        if self.return_tag_mask and self.category_by == 'item':
            self._cat_item_arr = [np.asarray(lst, dtype=np.int64) for lst in dataload.int_category_to_item_id]
            for cat_idx in range(self.eval_num_cats):
                logger.info(f'Category{cat_idx}={config["int_to_category"][cat_idx]} has #items={len(self._cat_item_arr[cat_idx])}')
            logger.info(f'category_counts: {dataload.category_counts}')
            assert self.eval_num_cats == len(self._cat_item_arr)

        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.device = config['device']

        self.text_keys = config['text_keys']
        self.processor = AutoProcessor.from_pretrained(config['item_pretrain_dir'])

        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']
        self.num_negatives = config['num_negatives']
        self.random_sample = config['pad_random_sample']
        if self.num_negatives:
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])  # for llm only
        else:
            if config['loss'] in ['nce', 'prior']:
                self.num_negatives = self.max_seq_length
        if not self.random_sample:
            logger.info(f"[Warning] --- Not using random samples for padding")
        logger.info(f"Text keys: {self.text_keys}")
        logger.info(f"Item prompt: {self.item_prompt}")
        self.chat_template = ("{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}"
                              "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}"
                              "{% endif %}{% if message['content'] is string %}{{ message['content'] }}{% else %}"
                              "{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' "
                              "in content or 'image_url' in content %}{% set image_count.value = image_count.value "
                              "+ 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
                              "<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or "
                              "'video' in content %}{% set video_count.value = video_count.value + 1 %}{% "
                              "if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|>"
                              "<|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}"
                              "{% endif %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}"
                              "<|im_start|>assistant\n{% endif %}")

    def __len__(self):
        return self.length

    def _neg_sample(self, item_set, tag_category=None, k=1):
        if tag_category is not None and self.rng.random() > self.neg_sample_mix_ratio:
            pool_arr = self._cat_item_arr[tag_category]
            weights = self.item_weights_by_cat[tag_category] if self.use_neg_sample_weights else None
        else:
            pool_arr = self._all_items_arr
            weights = self.item_interact_weights if self.use_neg_sample_weights else None

        if not isinstance(item_set, set):
            item_set = set(item_set)
            
        if not self.use_neg_sample_weights:
            draw_sz = min(pool_arr.size, k + len(item_set))
            draw = self.rng.choice(pool_arr, size=draw_sz, replace=False)
            if item_set:
                mask = ~np.isin(draw, list(item_set), assume_unique=False)
                out = draw[mask].tolist()
            else:
                out = draw.tolist()

            # Top-up at most once (rarely needed)
            need = k - len(out)
            if need > 0:
                extra = self.rng.choice(pool_arr, size=need, replace=False)
                # (collision with blacklist is extremely unlikely; if paranoid, filter again)
                out.extend(extra.tolist()[:need])
            return out[:k]

        res = []
        while len(res) < k:
            need = k - len(res)
            candidates = random.choices(pool_arr.tolist(), weights=weights, k=need)  # use weights=, not cum_weights=
            for x in candidates:
                if x not in item_set:
                    res.append(x)
        return res[:k]

    def _padding_sequence(self, item_seq, context_pad_len, pred_pad_len, data_type,
                          item_set=None, random_sample=False):
        # [pad, ..., pad, seq, ..., seq, pad, ..., pad]
        if random_sample:
            result = torch.empty(self.window_len, dtype=data_type)
            result[:context_pad_len] = torch.as_tensor(self._neg_sample(item_set, k=context_pad_len), dtype=data_type)
            result[context_pad_len:self.window_len - pred_pad_len] = item_seq
            result[self.window_len - pred_pad_len:] = torch.as_tensor(self._neg_sample(item_set, k=pred_pad_len), dtype=data_type)
        else:
            result = torch.zeros(self.window_len, dtype=data_type)
            result[context_pad_len:self.window_len-pred_pad_len] = item_seq
        return result

    def reconstruct_train_data(self, item_seq, context_pad_len, pred_pad_len):
        item_set = set(item_seq)
        item_seq = self._padding_sequence(torch.as_tensor(item_seq, dtype=torch.int64), context_pad_len, pred_pad_len,
                                          torch.int64, item_set=item_set, random_sample=self.random_sample)
        if self.neg_sample_by_cat:
            neg_item = [self._neg_sample(item_seq, cat_idx, k=self.num_negatives)
                        for cat_idx in range(self.eval_num_cats)]
            neg_item += [self._neg_sample(item_seq, k=self.num_negatives)]
        else:
            neg_item = [self._neg_sample(item_seq, k=self.num_negatives)]
        masked_index = torch.zeros(self.window_len, dtype=torch.int64)
        masked_index[context_pad_len:self.window_len-pred_pad_len] = 1  # use 0's to indicate padded positions
        return item_seq, neg_item, masked_index

    def _padding_time_sequence(self, sequence, context_pad_len, pred_pad_len):
        sequence = [0 for _ in range(context_pad_len)] + sequence + [0 for _ in range(pred_pad_len)]
        vq_time = []
        for time in sequence:
            dt = datetime.datetime.fromtimestamp(time, pytz.timezone('UTC'))
            vq_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])
        return torch.tensor(vq_time, dtype=torch.long)

    def process_event(self, event_seq, context_pad_len, pred_pad_len):
        event_mask = torch.zeros(self.window_len, self.eval_num_cats, dtype=torch.int64)
        event_categories = [[(i == idx) for i in range(self.eval_num_cats)]
                            for idx in event_seq[context_pad_len:self.window_len-pred_pad_len]]
        event_categories = torch.tensor(event_categories, dtype=torch.int64)
        event_mask[context_pad_len:self.window_len-pred_pad_len] = event_categories
        return event_mask

    def __getitem__(self, index):
        # everything has been converted to IDs instead of strings
        uid, context_end = self.dataload.valid_sample_locations[index]
        context_start = max(0, context_end - self.max_seq_length)
        context_pad_len = self.max_seq_length - context_end + context_start
        pred_len = min(self.dataload.train_seq_len[uid] - context_end, self.pred_len)
        item_seq = self.dataload.user_seq[uid][context_start:context_end + pred_len]  # python list
        item_seq, neg_item_seq, masked_index = self.reconstruct_train_data(item_seq, context_pad_len,
                                                                           self.pred_len - pred_len)
        # logger.info(f'item_seq: {item_seq}, neg_item: {neg_item[:20]}, masked_index: {masked_index}')
        if self.timestamp_required:
            time_seq = self.dataload.time_seq[uid][context_start:context_end + pred_len]
            time_seq = self._padding_time_sequence(time_seq, context_pad_len, self.pred_len - pred_len)
        else:
            time_seq = []

        pos_input_ids, pos_cu_input_lens, pos_position_ids, pos_pixel_values, pos_image_grid_thw = [], [], [], [], []
        pos_tag_categories = []

        def process_item(item_id_tensor, process_type="pos"):  # convert item_id to its text representation
            if isinstance(item_id_tensor, torch.Tensor):
                item_id = item_id_tensor.item()
            else:
                item_id = item_id_tensor
            item = self.dataload.item_to_info[item_id]
            text_str = ""
            tag_category = None
            if len(item) > 0:
                for key in self.text_keys:
                    value = item[key]
                    if value is not None and str(value) != 'nan':
                        text_str += f"{key.capitalize()}: {value}. "
                text_str = text_str.strip()

                if process_type == 'pos' and self.return_tag_mask and self.category_by == 'item':
                    tag_category = item['tag_category']

            else:
                item['image'] = None
                if process_type == 'pos' and self.return_tag_mask and self.category_by == 'item':
                    if item_id != 0:
                        logger.info(f"No information is found for item={self.item_list[item_id]}, id={item_id}")
                    tag_category = [False for _ in range(self.eval_num_cats)]

            if self.use_image and item['image'] is not None:
                if self.message_type in ['qwen']:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": item['image'],
                                    "resized_height": self.img_height,
                                    "resized_width": self.img_width,
                                },
                                {"type": "text",
                                 "text": f"Summarize item description into embedding: {text_str}"},
                            ],
                        }
                    ]
                elif self.message_type in ['llama']:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text",
                                 "text": f"Summarize item description into embedding: {text_str}"},
                                {"type": "image"}
                            ],
                        }
                    ]
                else:
                    raise ValueError(f"message_type={self.message_type} is undefined for messages with images")
                if self.has_chat_template:
                    text_prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                                     add_generation_prompt=True)
                else:
                    text_prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                                     add_generation_prompt=True,
                                                                     chat_template=self.chat_template)

                if self.message_type in ['qwen']:
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text_prompt],
                        images=image_inputs,
                        videos=video_inputs,
                        truncation=True,
                        max_length=self.max_text_length
                    )
                elif self.message_type in ['llama']:
                    if isinstance(item['image'], Image.Image):
                        raw_image = item['image']
                    elif item['image'].startswith("http://") or item['image'].startswith("https://"):
                        try:
                            raw_image = Image.open(requests.get(item['image'], stream=True).raw)
                            raw_image.load()
                        except:
                            # If opening/parsing the image fails, return a 64x64 black image
                            raw_image = Image.new("RGB", (64, 64), "black")
                    elif item['image'].startswith("file://"):
                        raw_image = Image.open(item['image'][7:])
                    elif item['image'].startswith("data:image"):
                        if "base64," in item['image']:
                            _, base64_data = item['image'].split("base64,", 1)
                            data = base64.b64decode(base64_data)
                            raw_image = Image.open(BytesIO(data))
                    else:
                        raw_image = Image.open(item['image'])
                    raw_image = raw_image.convert("RGB")
                    raw_image = raw_image.resize((self.img_width, self.img_height))
                    inputs = self.processor(
                        text=text_prompt,
                        images=raw_image,
                        truncation=True,
                        max_length=self.max_text_length
                    )
                else:
                    raise ValueError(f"message_type={self.message_type} is undefined for messages with images")
                ids = inputs['input_ids'][0]
                # mask = inputs['attention_mask'][0]
                pixel_values = inputs['pixel_values']  # shape: (16, 1176)
                if len(pixel_values) == 1:
                    pixel_values = pixel_values[0]  # (25, 3, 224, 224)
                if "image_sizes" in inputs:
                    image_grid_thw = inputs["image_sizes"]  # (644, 1031)
                else:
                    image_grid_thw = inputs["image_grid_thw"]
                return ids, pixel_values, image_grid_thw, tag_category
            else:
                if self.message_type in ["llama", "qwen"]:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text",
                                 "text": f"Summarize item description into embedding: {text_str}"},
                            ],
                        }
                    ]
                else:
                    raise ValueError(f"message_type={self.message_type} is undefined for messages with no images")

                if self.has_chat_template:
                    text_prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                                     add_generation_prompt=True)
                else:
                    text_prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                                     add_generation_prompt=True,
                                                                     chat_template=self.chat_template)
                inputs = self.processor(
                    text=[text_prompt],
                    truncation=True,
                    max_length=self.max_text_length,
                )
                ids = inputs['input_ids'][0]
                return ids, None, None, tag_category

        def get_item_tag(item_id_tensor, process_type="pos"):  # convert item_id to its text representation
            if isinstance(item_id_tensor, torch.Tensor):
                item_id = item_id_tensor.item()
            else:
                item_id = item_id_tensor
            item = self.dataload.item_to_info[item_id]
            tag_category = None
            if len(item) > 0:
                if process_type == 'pos' and self.return_tag_mask and self.category_by == 'item':
                    tag_category = item['tag_category']
            else:
                if process_type == 'pos' and self.return_tag_mask and self.category_by == 'item':
                    tag_category = [False for _ in range(self.eval_num_cats)]
            return tag_category

        if self.freeze_item_llm:
            for item_id in item_seq:  # process positive items
                tag_category = get_item_tag(item_id, "pos")
                if tag_category is not None:
                    pos_tag_categories.append(tag_category)
        else:
            for item_id in item_seq:  # process positive items
                ids, pixel_values, image_grid_thw, tag_category = process_item(item_id, "pos")
                # ids: list, pixel_values: np.ndarray, image_grid_thw: np.ndarray, tag_category: list
                pos_input_ids.extend(ids + [0 for _ in range(self.item_emb_token_n)])
                pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
                pos_position_ids.extend(
                    (torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())
                if pixel_values is not None:
                    pos_pixel_values.extend(pixel_values)
                    pos_image_grid_thw.extend(image_grid_thw)
                if tag_category is not None:
                    pos_tag_categories.append(tag_category)

        if self.category_by == 'event':
            event_seq = self.dataload.event_seq[uid][context_start:context_end + pred_len]
            event_seq = self._padding_sequence(torch.as_tensor(event_seq, dtype=torch.int64), context_pad_len,
                                                self.pred_len - pred_len, torch.int64)
            pos_tag_categories = self.process_event(event_seq, context_pad_len, self.pred_len - pred_len)

        # process negative items
        neg_dict = {}
        for cat_idx in range(len(neg_item_seq)):
            postfix = f'_cat{cat_idx}' if cat_idx < len(neg_item_seq) - 1 else ''
            neg_dict[f'neg_item_ids{postfix}'] = torch.as_tensor(neg_item_seq[cat_idx], dtype=torch.int64)  # torch.Size([8])
            if not self.freeze_item_llm:
                neg_input_ids, neg_cu_input_lens, neg_position_ids, neg_pixel_values, neg_image_grid_thw = [], [], [], [], []
                for neg_id in neg_item_seq[cat_idx]:
                    ids, pixel_values, image_grid_thw, _ = process_item(neg_id, "neg")
                    neg_input_ids.extend(ids + [0 for _ in range(self.item_emb_token_n)])
                    neg_cu_input_lens.append(len(ids) + self.item_emb_token_n)
                    neg_position_ids.extend(
                        (torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())
                    if pixel_values is not None:
                        neg_pixel_values.extend(pixel_values)
                        neg_image_grid_thw.extend(image_grid_thw)
                neg_dict[f'neg_input_ids{postfix}'] = torch.as_tensor(neg_input_ids, dtype=torch.int64)
                neg_dict[f'neg_cu_input_lens{postfix}'] = torch.as_tensor(neg_cu_input_lens, dtype=torch.int64)
                neg_dict[f'neg_position_ids{postfix}'] = torch.as_tensor(neg_position_ids, dtype=torch.int64)
                neg_dict[f'neg_pixel_values{postfix}'] = torch.as_tensor(np.array(neg_pixel_values), dtype=torch.float32)
                neg_dict[f'neg_image_grid_thw{postfix}'] = torch.as_tensor(np.array(neg_image_grid_thw), dtype=torch.int64)

        if self.freeze_item_llm:
            outputs = {
                "pos_item_ids": item_seq,  # torch.Size([51])
                "attention_mask": masked_index,
                "time_ids": torch.as_tensor(time_seq, dtype=torch.int64),
                "pos_tag_categories": torch.as_tensor(pos_tag_categories, dtype=torch.bool)
            }
        else:
            outputs = {
                "pos_item_ids": item_seq,  # torch.Size([51])
                "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
                "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
                "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64),
                "pos_pixel_values": torch.as_tensor(np.array(pos_pixel_values), dtype=torch.float32),
                "pos_image_grid_thw": torch.as_tensor(np.array(pos_image_grid_thw), dtype=torch.int64),
                "attention_mask": masked_index,
                "time_ids": torch.as_tensor(time_seq, dtype=torch.int64),
                "pos_tag_categories": torch.as_tensor(pos_tag_categories, dtype=torch.bool)
            }
        outputs.update(neg_dict)
        return outputs
