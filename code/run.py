import datetime
import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.allow_tf32 = True
from logging import getLogger
import json
from REC.data.dataload import InteractionData
from REC.data import *
import REC.data.comm as comm
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import torch.distributed as dist

import gc
import os
import argparse
import tempfile

torch.set_num_threads(1)          # intra-op
torch.set_num_interop_threads(1)  # inter-op


def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s


def run_loop(local_rank, config_file=None, saved=True, extra_args=[]):

    # configurations initialization
    config = Config(config_file_list=config_file)
    # config['debug'] = True

    device = torch.device("cuda", local_rank)
    config['device'] = device
    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                else:
                    config[key] = value
            except:
                raise ValueError(f"{key} {value} invalid")

    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    if not config.get("suppress_history", True):
        logger.info("You configure to not suppress seen items from history and it will be different from the paper.")

    # load item text information
    if 'text_path' in config:
        if os.path.isfile(os.path.join(config['text_path'], config['dataset'] + '.parquet')):
            config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.parquet')
        elif os.path.isfile(os.path.join(config['text_path'], config['dataset'] + '.csv')):
            config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.csv')
            raise ValueError(f'Support for CSV files has been deprecated. Use parquet instead.')
        else:
            raise ValueError(f'File {os.path.join(config["text_path"], config["dataset"])} not exist.')
        logger.info(f"Update text_path to {config['text_path']}")

    # fix configs
    if config['eval_pred_len'] not in config['metrics_pred_len_list']:
        config['metrics_pred_len_list'].append(config['eval_pred_len'])
    # Only add eval_pred_len // 2 if it's > 0 (to avoid negative indices after subtraction)
    half_pred_len = config['eval_pred_len'] // 2
    if half_pred_len > 0 and half_pred_len not in config['metrics_pred_len_list']:
        config['metrics_pred_len_list'].append(half_pred_len)
    assert all(isinstance(x, int) and x >= 0 for x in config['metrics_pred_len_list']), \
        "metrics_pred_len_list must be non-negative integers"
    config['metrics_pred_len_list'] = [x - 1 for x in config['metrics_pred_len_list']]
    config['metrics_pred_len_list'].sort()
    if config['loss'] not in ['prior'] or config['medusa_num_layers'] == 0:
        config['prior_switch'] = None
    if 'merrec' in config['dataset']:
        config['category_by'] = 'event'

    # get model and data
    logger.info("loading data, please be patient...")
    interaction_data = InteractionData(config)
    train_loader, valid_loader, test_loader = build_dataloader(config, interaction_data, log_detailed_results=config.get('log_detailed_results', False))
    logger.info(f"data loaded as {len(train_loader) = }")
    logger.info(f"{interaction_data=}")
    item_num = interaction_data.item_num
    user_num = interaction_data.user_num

    trainer = Trainer(config)
    logger.info(f"creating {config['model']} with {item_num=}, {user_num=}")
    if False:  # config['strategy'] == 'deepspeed':
        # not yet working
        logger.info(f"Use efficient model initialization with deepspeed")
        with trainer.strategy.module_init_context():
            model = get_model(config['model'])(config, interaction_data)
    else:
        # initialize the model on device to reduce CPU memory
        with torch.cuda.device(device):
            model = get_model(config['model'])(config, interaction_data).to(device)
    logger.info(f"{model=}")
    trainer.setup_model(model)
    del model
    gc.collect()

    world_size = torch.distributed.get_world_size()
    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    # synchronize before training begins
    torch.distributed.barrier()

    if config['val_only']:
        del valid_loader
        gc.collect()
        test_result = trainer.evaluate(test_loader, load_best_model=True, show_progress=config['show_progress'],
                                       init_model=True)
        del train_loader
        gc.collect()
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

    else:
        # training process
        best_valid_score, best_valid_result = trainer.fit(
            train_loader, valid_loader, saved=saved, show_progress=config['show_progress']
        )
        logger.info(f'Training Ended' + set_color('best valid ', 'yellow') + f': {best_valid_result}')
        del valid_loader
        gc.collect()

        # model evaluation
        test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=config['show_progress'])
        del train_loader
        gc.collect()

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str)
    args, extra_args = parser.parse_known_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp("matplotlib")
    os.environ['TRANSFORMERS_CACHE'] = tempfile.mkdtemp("huggingface")
    # optimize the memory: reduce the fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF") + ",expandable_segments:True"
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # disable lightning.fabric.launch
    os.environ['LT_CLI_USED'] = "1"
    # https://www.deepspeed.ai/tutorials/advanced-install/#cuda-version-mismatch
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"
    os.environ["LOGLEVEL"] = "INFO"
    config_file = args.config_file

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=30))

    # Setup the local process group (which contains ranks within the same machine)
    machine_rank = int(os.environ.get('GROUP_RANK', 0))
    num_machines = int(os.environ.get('GROUP_WORLD_SIZE', 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    assert comm._LOCAL_PROCESS_GROUP is None
    # initialize each group on every rank, but only assign appropriate group
    for i in range(num_machines):
        ranks_on_machine = list(range(local_world_size * i, local_world_size * i + local_world_size))
        pg = dist.new_group(ranks_on_machine)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg
    comm.synchronize()

    run_loop(local_rank=local_rank, config_file=config_file, extra_args=extra_args)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
