from torch.utils.data import Dataset

import os
import torch
import pandas as pd
import logging
from transformers import AutoProcessor
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import requests
from REC.data.qwen_vl_utils import process_vision_info
from REC.llm_dict import use_image_dict


class BatchItemDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.item_num = self.dataload.item_num
        self.item_list = dataload.id2token['item_id']
        self.config = config
        assert config['eval_num_cats'] > 1
        self.ordered_category = [config['int_to_category'][i] for i in range(config['eval_num_cats'])]
        self.category_by = config['category_by']
        self.device = config['device']
        self.logger = logging.getLogger()

    def __len__(self):
        return self.item_num

    def __getitem__(self, get_item_id):
        def process_item(item_id):
            item = self.dataload.item_to_info[item_id]
            if len(item) > 0:
                if self.category_by in ['event', 'user']:
                    tag_category = [True for _ in self.ordered_category]
                    original_tag_category = [True for _ in self.ordered_category]
                else:
                    tag_category = item['tag_category']
                    original_tag_category = item['original_tag_category']
            else:
                if item_id != 0:
                    self.logger.info(f"No information is found for item={self.item_list[item_id]}, id={item_id}")
                tag_category = [False for _ in self.ordered_category]
                original_tag_category = [False for _ in self.ordered_category]
            return tag_category, original_tag_category

        pos_tag_categories, pos_original_tag_categories = process_item(get_item_id)
        outputs = {
            "pos_tag_categories": torch.as_tensor(pos_tag_categories, dtype=torch.int64),
            "pos_original_tag_categories": torch.as_tensor(pos_original_tag_categories, dtype=torch.int64)
        }
        return outputs


class BatchTextDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.item_num = self.dataload.item_num
        self.item_list = dataload.id2token['item_id']
        self.config = config
        self.img_height = config['img_height']
        self.img_width = config['img_width']
        self.return_tag_mask = True if config["eval_num_cats"] > 1 else False
        if self.return_tag_mask:
            self.ordered_category = [config['int_to_category'][i] for i in range(config['eval_num_cats'])]
        else:
            self.ordered_category = []
        self.category_by = config['category_by']

        idx = config['item_pretrain_dir'].rfind('/')
        if idx == -1:
            item_llm_name = config['item_pretrain_dir']
        else:
            item_llm_name = config['item_pretrain_dir'][idx + 1:]

        can_use_img =  use_image_dict[item_llm_name]['use_image']
        self.use_image = config['use_image']
        if self.use_image:
            assert can_use_img, f"item_llm={item_llm_name} doesn't support images..."
        self.use_image_online = config['use_image_online']
        self.message_type = use_image_dict[item_llm_name]['message_type']
        self.has_chat_template = use_image_dict[item_llm_name]['has_chat_template']

        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.device = config['device']

        self.text_path = config['text_path']
        self.text_keys = config['text_keys']
        self.processor = AutoProcessor.from_pretrained(config['item_pretrain_dir'])
        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']
        self.logger = logging.getLogger()

    def __len__(self):
        return self.item_num

    def __getitem__(self, get_item_id):
        def process_item(item_id):
            chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{% endif %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            item = self.dataload.item_to_info[item_id]
            text_str = ""
            tag_category = []
            original_tag_category = []
            if len(item) > 0:
                for key in self.text_keys:
                    value = item[key]
                    if value is not None and str(value) != 'nan':
                        text_str += f"{key.capitalize()}: {value}. "
                text_str = text_str.strip()

                if self.category_by == 'event':
                    tag_category = [True for _ in self.ordered_category]
                    original_tag_category = [True for _ in self.ordered_category]
                else:
                    tag_category = item['tag_category']
                    original_tag_category = item['original_tag_category']
            else:
                item['image'] = None
                if self.return_tag_mask:
                    if item_id != 0:
                        self.logger.info(f"No information is found for item={self.item_list[item_id]}, id={item_id}")
                    tag_category = [False for _ in self.ordered_category]
                    original_tag_category = [False for _ in self.ordered_category]
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
                                                                     chat_template=chat_template)

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
                    pixel_values = pixel_values[0]
                if "image_sizes" in inputs:
                    image_grid_thw = inputs["image_sizes"]
                else:
                    image_grid_thw = inputs["image_grid_thw"]
                return ids, pixel_values, image_grid_thw, tag_category, original_tag_category
            else:
                if self.message_type in ["qwen"]:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text",
                                 "text": f"Summarize item description into embedding: {text_str}"},
                            ],
                        }
                    ]
                elif self.message_type in ["llama"]:
                    messages = [
                        {
                            "role": "user",
                            "content": f"Summarize item description into embedding: {text_str}"
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
                                                                     chat_template=chat_template)
                inputs = self.processor(
                    text=[text_prompt],
                    truncation=True,
                    max_length=self.max_text_length,
                )
                ids = inputs['input_ids'][0]
                return ids, None, None, tag_category, original_tag_category

        pos_input_ids, pos_cu_input_lens, pos_position_ids, pos_pixel_values, pos_image_grid_thw = [], [], [], [], []
        ids, pixel_values, image_grid_thw, pos_tag_categories, pos_original_tag_categories = process_item(get_item_id)
        pos_input_ids.extend(ids + [0 for _ in range(self.item_emb_token_n)])
        pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
        pos_position_ids.extend(
            (torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())
        if pixel_values is not None:
            pos_pixel_values.extend(pixel_values)
            pos_image_grid_thw.extend(image_grid_thw)
        outputs = {
            "pos_item_ids": torch.as_tensor(get_item_id, dtype=torch.int64),
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64),
            "pos_pixel_values": torch.as_tensor(np.array(pos_pixel_values), dtype=torch.float32),
            "pos_image_grid_thw": torch.as_tensor(np.array(pos_image_grid_thw), dtype=torch.int64),
            "pos_tag_categories": torch.as_tensor(pos_tag_categories, dtype=torch.int64),
            "pos_original_tag_categories": torch.as_tensor(pos_original_tag_categories, dtype=torch.int64)
        }
        return outputs
