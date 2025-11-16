import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate
import re
try:
    from torch._six import string_classes
except:
    string_classes = str

import collections

np_str_obj_array_pattern = re.compile(r"[SaUO]")

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def customize_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # batch here is a list of tensors, where the number of elements in the list is the b_sz
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            out = torch.empty((len(batch),) + elem.shape, dtype=elem.dtype, device=elem.device)
            out.share_memory_()  # Move to shared memory if necessary
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: customize_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        return batch


def seq_eval_collate(batch):
    user_ids = []
    item_seq = []
    time_seq = []
    history_i = []
    item_target = []
    target_tag_categories = []
    outlier_users = []

    for item in batch:
        user_ids.append(item[0])
        item_seq.append(item[2])
        history_i.append(item[1])
        item_target.append(item[3])
        time_seq.append(item[4])
        target_tag_categories.append(item[5])
        outlier_users.append(item[6])

    history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_i)])
    history_i = torch.cat(history_i)

    user_ids = torch.tensor(user_ids, dtype=torch.int64)  # [batch]
    item_seq = torch.stack(item_seq)  # [batch, len]
    item_target = torch.stack(item_target)  # [batch, pred_len]
    time_seq = torch.tensor(time_seq)  # [batch]
    target_tag_seq = torch.tensor(target_tag_categories)
    positive_u = torch.arange(item_seq.shape[0]).unsqueeze(-1).repeat(1, item_target.shape[1])  # [batch, pred_len]
    # both history_u and positive_u is not the user_id. It is the reindexed version: arange(batch_size)
    outlier_users = torch.tensor(outlier_users, dtype=torch.bool)

    return user_ids, item_seq, item_target, (history_u, history_i), positive_u, time_seq, target_tag_seq, \
        outlier_users


def customize_rmpad_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):  # this is called
        output = {}
        for key in elem:
            vals = [d[key] for d in batch]  # build once
            if any(['_input_ids' in key, '_cu_input_lens' in key, '_position_ids' in key,
                    '_pixel_values' in key, '_image_grid_thw' in key, 'time_ids' in key]):
                # Preallocate once and copy slices (works for 1-D or N-D)
                output[key] = torch.cat(vals, dim=0)
            else:
                output[key] = customize_collate(vals)

        return output
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        return batch


def _concat_prealloc(vals):
    """Concatenate along dim=0 by preallocating once."""
    base = vals[0]
    total0 = sum(v.shape[0] for v in vals)
    out = base.new_empty((total0, *base.shape[1:]))  # keeps dtype/device
    i = 0
    for v in vals:
        n = v.shape[0]
        out[i:i+n].copy_(v)
        i += n
    return out
