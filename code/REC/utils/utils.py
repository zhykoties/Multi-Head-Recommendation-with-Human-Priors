# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT

import datetime
import importlib
import os
import random

import numpy as np
from typing import Union, List, Dict
import torch
from tensorboardX import SummaryWriter


def get_local_time():
    r"""Get current time with microseconds for higher precision.

    Returns:
        str: current time
    """
    # Using microseconds reduces the chance of collision significantly.
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S-%f')
    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    # Use exist_ok=True to prevent race conditions.
    os.makedirs(dir_path, exist_ok=True)


def get_model(model_name):

    model_file_name = model_name.lower()
    model_module = None

    module_path = '.'.join(['REC.model.IDNet', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    if model_module is None:
        module_path = '.'.join(['REC.model.HLLM', model_file_name])

    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))

    model_class = getattr(model_module, model_name)
    return model_class


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value >= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value <= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, eval_pred_len, valid_metric=None):
    r""" return valid score from valid result

    Args:
        valid_result (dict): valid result
        eval_pred_len (int): prediction length during evaluation
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        if f'pred_{eval_pred_len - 1}' in valid_result:
            return valid_result[f'pred_{eval_pred_len - 1}'][valid_metric]
        else:
            return valid_result[valid_metric]
    else:
        if f'pred_{eval_pred_len - 1}' in valid_result:
            return valid_result[f'pred_{eval_pred_len - 1}']['Recall@10']
        else:
            return valid_result['Recall@10']


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return '    '.join([str(metric) + ': ' + str(value) for metric, value in result_dict.items()])


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_tensorboard(logger):
    r""" Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for 
    visualization within the TensorBoard UI.
    For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

    Args:
        logger: its output filename is used to name the SummaryWriter's log_dir.
                If the filename is not available, we will name the log_dir according to the current time.

    Returns:
        SummaryWriter: it will write out events and summaries to the event file.
    """
    base_path = 'log_tensorboard'

    dir_name = None
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            # Use the logger's filename as the directory name
            dir_name = os.path.basename(getattr(handler, 'baseFilename')).split('.')[0]
            break
            
    if dir_name is None:
        # Fallback to a time-based name if logger filename is not found
        dir_name = '{}-{}'.format('model', get_local_time())

    dir_path = os.path.join(base_path, dir_name)

    # --- Start of Robust Directory Creation Logic ---
    # Check if we are in a distributed environment
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    
    # Only rank 0 should create the directory
    if not is_distributed or torch.distributed.get_rank() == 0:
        ensure_dir(dir_path)

    # All other processes should wait until the directory is created
    if is_distributed:
        torch.distributed.barrier()
    # --- End of Robust Directory Creation Logic ---

    writer = SummaryWriter(dir_path)
    return writer


def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)


def save_log_dict(data_dict, prefix, folder_path):
    """
    Saves a dict whose values are either numpy arrays or lists of strings:
      - numpy arrays → uncompressed .npy (very fast)
      - lists of strings → newline‑delimited .txt (also very fast)
    """
    os.makedirs(folder_path, exist_ok=True)

    for key, val in data_dict.items():
        out_path = os.path.join(folder_path, f'{prefix}-{key}')

        if isinstance(val, np.ndarray):
            # store array as uncompressed .npy
            np.save(out_path + ".npy", val, allow_pickle=False)

        elif isinstance(val, list):
            # 1-D list of strings
            if all(isinstance(x, str) for x in val):
                with open(out_path + ".txt", "w", encoding="utf-8") as f:
                    for s in val:
                        f.write(s + "\n")
            # 2-D list of strings
            elif all(isinstance(x, list) for x in val) and all(isinstance(y, str) for row in val for y in row):
                with open(out_path + ".txt", "w", encoding="utf-8") as f:
                    for row in val:
                        # join inner list items with commas
                        f.write(",".join(row) + "\n")
            else:
                np.save(out_path + ".npy", np.array(val), allow_pickle=False)

        else:
            raise ValueError(f"Unsupported type for key {key}: {type(val)}")


def load_log_dict(prefix: str, folder_path: str) -> Dict[str, Union[np.ndarray, List[str], List[List[str]]]]:
    """
    Loads everything saved by save_log_dict for a given prefix.
    Returns a dict mapping each key (e.g. 'head-source', 'item_history', …) to either:
      - a numpy array (for .npy files)
      - a list of strings (for 1-D .txt)
      - a list of list of strings (for comma‑delimited 2-D .txt)
    """
    data_dict = {}
    # Look at every file in the folder
    for fn in os.listdir(folder_path):
        # we're only interested in files that start with "prefix-"
        if not fn.startswith(prefix + "--"):
            continue

        key_part, ext = os.path.splitext(fn[len(prefix) + 2:])
        full_path = os.path.join(folder_path, fn)

        if ext == ".npy":
            data_dict[key_part] = np.load(full_path, allow_pickle=True)

        elif ext == ".txt":
            with open(full_path, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f]
            if any("," in line for line in lines):
                data_dict[key_part] = [line.split(",") for line in lines]
            else:
                data_dict[key_part] = lines

        else:
            continue

    # sanity check: you should have exactly these five keys
    expected = {"head_source", "item_history", "item_tgt", "user", "values", "recommend_items"}
    missing = expected - set(data_dict)
    if missing:
        raise RuntimeError(f"Missing files for keys: {missing}")
    return data_dict
