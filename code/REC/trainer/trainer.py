from collections import defaultdict

import copy
import json
import gc
import os
import sys
import tempfile
from logging import getLogger
from time import time
import time as t
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import wandb
from tqdm import tqdm
import deepspeed

from REC.data.dataset import BatchItemDataset, BatchTextDataset
from REC.data.dataset.collate_fn import customize_rmpad_collate
from torch.utils.data import DataLoader
from REC.evaluator import Evaluator, Collector
from REC.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
    save_log_dict,
)
from REC.utils.lr_scheduler import *
from REC.utils.wandblogger import name_with_datetime

from lightning.fabric.fabric import Fabric
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy

import pandas as pd
import gzip, json
from typing import Iterable, List


def save_strings(strings: Iterable[str], path: str) -> None:
    # Streams; doesn't build a giant JSON blob in memory
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for s in strings:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.eval_pred_len = self.config["eval_pred_len"]
        self.metrics_pred_len_list = self.config["metrics_pred_len_list"]
        self.config.recorded_config = {
            "seed": self.config["seed"],
            "dataset": self.config["dataset"],
            "loss": self.config["loss"],
            "total_iters": self.config["total_iters"],
            "eval_freq": self.config["eval_freq"],
            "train_batch_size": self.config["train_batch_size"],
            "learning_rate": self.config["optim_args"]["learning_rate"],
            "weight_decay": self.config["optim_args"]["weight_decay"],
            "lr_warmup": self.config["scheduler_args"]["warmup"],
            "medusa_lambda": self.config["medusa_lambda"],
            "medusa_num_layers": self.config["medusa_num_layers"],
            "num_segment_head": self.config.get("num_segment_head", 1),
            "num_prior_head": self.config.get("num_prior_head", 1),
            "head_interaction": self.config["head_interaction"],
            "eval_num_cats": self.config["eval_num_cats"],
            "neg_sample_by_cat": self.config["neg_sample_by_cat"],
            "max_text_length": self.config["MAX_TEXT_LENGTH"],
            "max_item_list_length": self.config["MAX_ITEM_LIST_LENGTH"],
            "item_pretrain_dir": self.config["item_pretrain_dir"],
            "user_pretrain_dir": self.config["user_pretrain_dir"],
            "pred_len": self.config["pred_len"],
            "subset_user": self.config.get("subset_user", False),
            "subset_user_div": self.config.get("subset_user_div", 1),
            "subset_user_rmd": self.config.get("subset_user_rmd", 0),
            "head_interaction": self.config.get("head_interaction", "multiplicative"),
            "neg_sample_by_cat": self.config.get("neg_sample_by_cat", False),
            "weighted_prior_loss": self.config.get("weighted_prior_loss", False),
            "tag_version": self.config.get("tag_version", "v1"),
            "min_seq_len": self.config.get("min_seq_len", 10),
            "prior_given_at_test": self.config.get("prior_given_at_test", False),
            "given_prior_len": self.config.get("given_prior_len", 1),
            "outlier_user_metrics": self.config.get("outlier_user_metrics", "category"),
        }
        self.logger = getLogger()

        self.optim_args = config["optim_args"]
        self.stopping_step = config["stopping_step"]
        self.clip_grad_norm = config.get("clip_grad_norm", 1.0)
        self.valid_metric = config["valid_metric"].lower()
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.gpu_available = torch.cuda.is_available() and config["use_gpu"]
        self.device = config["device"]
        if self.config['category_by'] == 'user':
            assert self.config['prior_given_at_test'] is True and self.config['given_prior_len'] == 1   

        self.rank = torch.distributed.get_rank()
        self.saved_model_name = (
            f'{self.config["model"]}-{self.config["dataset"]}'
            f'-{self.config["save_model_note"]}.pth'
        )
        self.wandblogger = WandbLogger(
            config, self.saved_model_name[:-4]
        )  # -4 to remove .pth
        self.load_checkpoint_name = config["load_checkpoint_name"]
        self.freeze_item_llm = config.get('freeze_item_llm', False)
        if config.load_checkpoint_name is not None and not self.freeze_item_llm:
            self.saved_model_name = config.load_checkpoint_name
        if self.rank == 0:
            self.tensorboard = get_tensorboard(self.logger)
            self.results_df = pd.DataFrame(
                columns=[
                    "Pred Metrics",
                    "Train Steps",
                    "recall@5",
                    "recall@10",
                    "recall@50",
                    "recall@200",
                    "ndcg@5",
                    "ndcg@10",
                    "ndcg@50",
                    "ndcg@200",
                ]
            )

        self.checkpoint_dir = config["checkpoint_dir"]
        if self.rank == 0:
            ensure_dir(self.checkpoint_dir)

        self.saved_model_file = os.path.join(self.checkpoint_dir, self.saved_model_name)
        if self.load_checkpoint_name is not None:
            self.load_checkpoint_name = os.path.join(self.checkpoint_dir, self.load_checkpoint_name)

        self.use_text = config["use_text"]
        self.total_iters = config["total_iters"]
        self.start_iter = 0
        self.train_step = self.start_iter
        self.accumulate_grad = self.config.get("accumulate_grad", 1)
        self.no_improve_times = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.update_interval = (
            config["update_interval"] if config["update_interval"] else 20
        )
        self.eval_interval = config["eval_interval"]
        self.scheduler_config = config["scheduler_args"]

        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_feature, self.all_item_tags = None, None
        self.tot_item_num = None
        self.log_detailed_results = config.get("log_detailed_results", False)
        self.log_wandb = config.log_wandb
        self.eval_by_cat = config.get("eval_by_cat", True)
        self.prior_switch = config.get("prior_switch", None)
        self.int_to_category = config["int_to_category"]
        self.eval_num_cats = config['eval_num_cats']
        self.outlier_user_metrics = config['outlier_user_metrics']
        self.save_for_eval = config.get("save_for_eval", False)
        if self.save_for_eval:
            assert self.config["val_only"]
        self._setup_fabric()

    def setup_model(self, model: torch.nn.Module) -> None:
        if self.freeze_item_llm:
            assert self.config['model'] == 'HLLM'

        # set up model for train or inference
        self.model = model

        # set up learnable parameter
        self.logger.info("set up learnable parameter")
        
        if self.config['freeze_prefix'] or self.freeze_item_llm:
            freeze_prefix = self.config['freeze_prefix'] if self.config['freeze_prefix'] else []
            if self.freeze_item_llm:
                freeze_prefix.extend(['item_llm', 'item_emb_tokens'])
            self._freeze_params(freeze_prefix)

        for n, p in self.model.named_parameters():
            self.logger.info(f"{n} {p.size()=} {p.requires_grad=} {p.device=}")

        # set up optimizer
        self.logger.info("set up optimizer")
        self.optimizer = self._build_optimizer()

    def _freeze_params(self, freeze_prefix):
        for name, param in self.model.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    param.requires_grad = False

    def _unfreeze_params(self, unfreeze_prefix):
        for name, param in self.model.named_parameters():
            for prefix in unfreeze_prefix:
                if name.startswith(prefix):
                    param.requires_grad = True

    def _build_scheduler(self, warmup_steps=None, tot_steps=None):
        if self.scheduler_config["type"] == "cosine":
            self.logger.info(
                f"Use cosine scheduler with {warmup_steps} warmup {tot_steps} total steps"
            )
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        elif self.scheduler_config["type"] == "linear":
            self.logger.info(
                f"Use linear scheduler with {warmup_steps} warmup {tot_steps} total steps"
            )
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        else:
            self.logger.info(f"Use constant scheduler")
            return get_constant_schedule(self.optimizer)

    def _build_optimizer(self):
        if len(self.optim_args) == 4:
            params = self.model.named_parameters()
            modal_params = []
            recsys_params = []
            modal_decay_params = []
            recsys_decay_params = []
            decay_check_name = self.config["decay_check_name"]
            for index, (name, param) in enumerate(params):
                if param.requires_grad:
                    if "visual_encoder" in name:
                        modal_params.append(param)
                    else:
                        recsys_params.append(param)
                    if decay_check_name:
                        if decay_check_name in name:
                            modal_decay_params.append(param)
                        else:
                            recsys_decay_params.append(param)
            if decay_check_name:
                optimizer = optim.AdamW([
                    {"params": modal_decay_params, "lr": self.optim_args["modal_lr"],
                     "weight_decay": self.optim_args["modal_decay"]},
                    {"params": recsys_decay_params, "lr": self.optim_args["rec_lr"],
                     "weight_decay": self.optim_args["rec_decay"]}
                ])
                optim_output = set_color(
                    f"recsys_decay_params_len: {len(recsys_decay_params)}  modal_params_decay_len: {len(modal_decay_params)}",
                    "blue",
                )
                self.logger.info(optim_output)
            else:
                optimizer = optim.AdamW([
                    {"params": modal_params, "lr": self.optim_args["modal_lr"],
                     "weight_decay": self.optim_args["modal_decay"]},
                    {"params": recsys_params, "lr": self.optim_args["rec_lr"],
                     "weight_decay": self.optim_args["rec_decay"]}
                ])
                optim_output = set_color(
                    f"recsys_lr_params_len: {len(recsys_params)}  modal_lr_params_len: {len(modal_params)}",
                    "blue",
                )
                self.logger.info(optim_output)
        elif self.config["lr_mult_prefix"] and self.config["lr_mult_rate"]:
            normal_params_dict = {
                "params": [],
                "lr": self.optim_args["learning_rate"],
                "weight_decay": self.optim_args["weight_decay"],
            }
            high_lr_params_dict = {
                "params": [],
                "lr": self.optim_args["learning_rate"] * self.config["lr_mult_rate"],
                "weight_decay": self.optim_args["weight_decay"],
            }
            self.logger.info(
                f'Use higher lr rate {self.config["lr_mult_rate"]} x {self.optim_args["learning_rate"]} for prefix {self.config["lr_mult_prefix"]}'
            )

            for n, p in self.model.named_parameters():
                if any(n.startswith(x) for x in self.config["lr_mult_prefix"]):
                    self.logger.info(
                        f"high lr param: {n} {self.optim_args['learning_rate'] * self.config['lr_mult_rate']}"
                    )
                    high_lr_params_dict["params"].append(p)
                else:
                    normal_params_dict["params"].append(p)
            optimizer = optim.AdamW([normal_params_dict, high_lr_params_dict])
        elif self.config['strategy'] == 'deepspeed':
            params = self.model.parameters()
            optimizer = deepspeed.ops.adam.fused_adam.FusedAdam(params, lr=self.optim_args['learning_rate'], weight_decay=self.optim_args['weight_decay'])
            self.logger.info("using DeepSpeedCPUAdam due to deepspeed strategy")
        else:
            optimizer = optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                    lr = self.optim_args['learning_rate'],
                                    weight_decay = self.optim_args['weight_decay'])
        return optimizer

    def _valid_iter(self, valid_data, show_progress=False):
        if self.config["debug"]:
            result_summary = dict()
            for pred_idx in self.metrics_pred_len_list:
                result_summary[f"pred_{pred_idx}"] = {"None": 0}
            return 0, result_summary

        torch.distributed.barrier()
        valid_result = self.evaluate(
            valid_data, load_best_model=False, show_progress=show_progress
        )
        del valid_data
        valid_score = calculate_valid_score(valid_result, self.eval_pred_len, self.valid_metric)
        gc.collect()
        torch.distributed.barrier()
        return valid_score, valid_result

    def _save_checkpoint(self, iter_idx, verbose=True):
        r"""Store the model parameters information and training information.
        Args:
            iter_idx (int): the current training iterations
        """

        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "config": self.config,
            "iter_idx": iter_idx,
            "best_valid_score": self.best_valid_score,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
        }

        self.lite.save(
            os.path.join(self.checkpoint_dir, self.saved_model_name), state=state
        )

        if self.rank == 0 and verbose:
            self.logger.info(set_color("Saving current", "blue") + f": {self.saved_model_file}")

    def _resume_from_checkpoint(self, checkpoint_path):
        self.logger.info(f"---> Resume training is enabled. Loading from {checkpoint_path}")
        if self.freeze_item_llm:
            shard_dir = os.path.join(checkpoint_path, "full_model_fp32.pt")
            if os.path.isfile(os.path.join(shard_dir, "pytorch_model.bin")):
                state = torch.load(os.path.join(shard_dir, "pytorch_model.bin"), map_location="cpu")
                self.model.load_state_dict(state)
            else:
                with open(os.path.join(shard_dir, "pytorch_model.bin.index.json")) as f:
                    index = json.load(f)

                state = {}
                for shard_file in tqdm(sorted(set(index["weight_map"].values()))):
                    state.update(torch.load(os.path.join(shard_dir, shard_file), map_location="cpu"))  # fp32

                self.model.load_state_dict(state)
        else:
            state = {
                "model": self.model,
                "optimizer": self.optimizer,
            }
            state = self.lite.load(checkpoint_path, state)
            torch.set_rng_state(state['rng_state'])
            torch.cuda.set_rng_state(state['cuda_rng_state'])
        # the model and optimizer should be already loaded inplace
        torch.distributed.barrier()
        del state
        gc.collect()

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _generate_train_loss_output(self, iter_idx, s_time, e_time, losses):
        des = self.config["loss_decimal_place"] or 4
        train_loss_output = (set_color("Iter %d training", "green") + " [" + set_color("time", "blue") +
                             ": %.2fs, ") % (iter_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = set_color("train_loss%d", "blue") + ": %." + str(des) + "f"
            train_loss_output += ", ".join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = "%." + str(des) + "f"
            train_loss_output += set_color("train loss", "blue") + ": " + des % losses
        del iter_idx, s_time, e_time, losses, des
        return train_loss_output + "]"

    def _add_train_loss_to_tensorboard(self, iter_idx, losses, tag="Loss/Train"):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, iter_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, iter_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            "learning_rate": self.config["learning_rate"],
            "weight_decay": self.config["weight_decay"],
            "train_batch_size": self.config["train_batch_size"],
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter for parameters in self.config.parameters.values() for parameter in parameters
        }.union({"model", "dataset", "config_files", "device"})
        # other model-specific hparam
        hparam_dict.update({
            para: val for para, val in self.config.final_config_dict.items()
            if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            k = k.replace("@", "_")
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {"hparam/best_valid_result": best_valid_result})

    def to_device(self, data, non_blocking=True):
        if torch.is_tensor(data):
            # async H2D copy if (a) src is CPU, (b) DataLoader used pin_memory=True
            return data.to(self.device, non_blocking=non_blocking)

        # preserve structure
        if isinstance(data, dict):
            return {k: self.to_device(v, non_blocking) for k, v in data.items()}
        if isinstance(data, list):
            return [self.to_device(v, non_blocking) for v in data]
        if isinstance(data, tuple):
            return tuple(self.to_device(v, non_blocking) for v in data)

        # e.g., ints/strings/None
        return data

    def _setup_fabric(self) -> None:
        world_size, local_world_size = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
        nnodes = world_size // local_world_size
        assert nnodes == int(os.environ.get('GROUP_WORLD_SIZE', 1)), f"inconsistent {nnodes=} != GROUP_WORLD_SIZE={os.environ.get('GROUP_WORLD_SIZE', 1)}"
        precision = self.config['precision'] if self.config['precision'] else '32'
        if self.config['strategy'] == 'deepspeed':
            self.logger.info(f"Use deepspeed strategy")
            self.strategy = DeepSpeedStrategy(stage=self.config["stage"], precision=precision)
            self.lite = Fabric(accelerator='gpu', strategy=self.strategy, precision=precision, num_nodes=nnodes, loggers=self.logger)
        else:
            self.logger.info(f"Use DDP strategy")
            self.strategy = DDPStrategy(find_unused_parameters=True)
            self.lite = Fabric(accelerator='gpu', strategy=self.strategy, precision=precision, num_nodes=nnodes, loggers=self.logger)
        # The launch() method should only be used if you intend to specify accelerator, devices, and so on in the code (programmatically).
        # If you are launching with the Lightning CLI, fabric run ..., remove launch() from your code.
        if not bool(int(os.environ.get("LT_CLI_USED", "0"))):
            self.logger.info(f"detected it is not launched from lightning CLI, will need fabric.launch")
            self.lite.launch()
        else:
            self.logger.info(f"detected it is launched from lightning CLI, will by pass fabric.launch")

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.scheduler_config:
            warmup_rate = self.scheduler_config.get("warmup", 0.001)
            warmup_steps = self.total_iters * warmup_rate
            self.lr_scheduler = self._build_scheduler(warmup_steps=warmup_steps, tot_steps=self.total_iters)

        self.logger.info("set up fabric with model and optimizer")
        self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)

        if self.load_checkpoint_name is not None:
            self._resume_from_checkpoint(self.load_checkpoint_name)
            if self.freeze_item_llm:
                self.logger.info('Frozen item_llm. Start computing item features...')
                self.compute_item_feature(self.config, valid_data.dataset.dataload)
                self.model.set_all_item_embeds(self.item_feature)
                self.logger.info("Finished computing item features...")

        for n, p in self.model.named_parameters():
            self.logger.info(f"{n} {p.size()} {p.requires_grad}")

        epoch_idx = 0
        train_data.sampler.set_epoch(epoch_idx)
        iterator = iter(train_data)
        self.model.train()
        total_loss = 0
        loss_running_avg = 0
        training_start_time = time()
        if self.rank == 0:
            pbar = tqdm(
                total=self.total_iters * self.accumulate_grad,
                bar_format="{l_bar}{bar}\n{r_bar}",
                miniters=self.update_interval,
                desc=set_color(f"Train [{epoch_idx:>2}]", "pink"),
                file=sys.stdout,
                ascii=True,
                initial=self.start_iter * self.accumulate_grad,
            )
        bwd_time = t.time()

        for acc_iter_idx in range(
            self.start_iter * self.accumulate_grad, self.total_iters * self.accumulate_grad,
        ):
            try:  # https://stackoverflow.com/a/58876890/8365622
                data = next(iterator)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                epoch_idx += 1
                train_data.sampler.set_epoch(epoch_idx)
                self.logger.info(f"Restarting iterator for epoch {epoch_idx} at iter {acc_iter_idx}")
                iterator = iter(train_data)
                data = next(iterator)

            start_time = bwd_time
            data = self.to_device(data, non_blocking=True)

            data_time = t.time()
            model_out = self.model(data)
            del data

            fwd_time = t.time()

            losses = model_out.pop("loss")
            self._check_nan(losses)
            total_loss += losses.item()
            loss_running_avg += losses.item()

            self.lite.backward(losses / self.accumulate_grad)
            if hasattr(self.model, "item_emb_tokens") and acc_iter_idx % 200 == 0:
                if self.model.item_emb_tokens.grad is None:
                    self.logger.info("emb_tokens is not getting grad")

            del losses

            # Only update the optimizer every `self.accumulate_grad` steps
            if (acc_iter_idx + 1) % self.accumulate_grad == 0:
                loss_running_avg /= self.accumulate_grad
                self.train_step += 1
                grad_norm = self.optimizer.step()
                self.optimizer.zero_grad()  # Reset gradients after accumulation step

                if self.scheduler_config:
                    self.lr_scheduler.step()

                # logging for training
                if show_progress and self.train_step % self.update_interval == 0:
                    bwd_time = t.time()
                    msg = (
                        f"loss: {loss_running_avg:.4f} data: {data_time - start_time:.3f} "
                        f"fwd: {fwd_time - data_time:.3f} bwd: {bwd_time - fwd_time:.3f}"
                    )
                    if self.scheduler_config:
                        msg = f"lr: {self.lr_scheduler.get_lr()[0]:.7f} " + msg
                    for k, v in model_out.items():
                        if isinstance(v, torch.Tensor):
                            msg += f" {k}: {v.item():.3f}"
                        else:
                            msg += f" {k}: {v:.3f}"
                    if grad_norm:
                        msg += f" grad_norm: {grad_norm.sum():.4f}"
                    self.wandblogger.log_metrics(
                        {
                            **{
                                "train_loss": loss_running_avg,
                                "learning_rate": self.lr_scheduler.get_lr()[0],
                            },
                            **model_out,
                        },
                        step=self.train_step, head="train",
                    )
                    if self.rank == 0:
                        pbar.set_postfix_str(msg, refresh=False)
                        pbar.update(self.update_interval)
                        self.logger.info("\n" + "-" * 50)

                loss_running_avg = 0

                if self.config["debug"] and self.train_step >= 10:
                    break

                if self.train_step % self.eval_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    # logging for validation
                    train_loss = total_loss / self.accumulate_grad
                    self.train_loss_dict[self.train_step] = (
                        sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                    )
                    training_end_time = time()
                    train_loss_output = self._generate_train_loss_output(
                        self.train_step,
                        training_start_time,
                        training_end_time,
                        train_loss,
                    )
                    if verbose:
                        self.logger.info(train_loss_output)
                    if self.rank == 0:
                        self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

                    if self.eval_interval <= 0 or not valid_data:
                        if saved:
                            self._save_checkpoint(self.train_step, verbose=verbose)
                        continue

                    # evaluation on validation set
                    valid_start_time = time()
                    valid_score, valid_result = self._valid_iter(
                        valid_data, show_progress=show_progress
                    )
                    self.model.train()
                    (
                        self.best_valid_score,
                        self.no_improve_times,
                        stop_flag,
                        update_flag,
                    ) = early_stopping(
                        valid_score,
                        self.best_valid_score,
                        self.no_improve_times,
                        max_step=self.stopping_step,
                        bigger=self.valid_metric_bigger,
                    )
                    valid_end_time = time()
                    valid_score_output = (
                        set_color("Step %d evaluating", "green")
                        + " ["
                        + set_color("time", "blue")
                        + ": %.2fs, "
                        + set_color("valid_score", "blue")
                        + ": %f]"
                    ) % (
                        self.train_step,
                        valid_end_time - valid_start_time,
                        valid_score,
                    )
                    valid_result_output = set_color("valid result", "blue") + ": \n"
                    for pred_idx in self.metrics_pred_len_list:
                        valid_result_output += (
                            f"pred_{pred_idx} --> {dict2str(valid_result[f'pred_{pred_idx}'])}\n"
                        )
                    for pred_idx in self.metrics_pred_len_list:
                        self.wandblogger.log_metrics(
                            valid_result[f"pred_{pred_idx}"],
                            step=self.train_step, head=f"valid_pred_{pred_idx}",
                        )
                    if 'shared' in valid_result:
                        valid_result_output += (
                            f"Shared --> {dict2str(valid_result['shared'])} \n "
                        )
                        self.wandblogger.log_metrics(
                            valid_result['shared'], step=self.train_step, head=f"Shared",
                        )
                    if verbose:
                        self.logger.info(valid_score_output)
                        self.logger.info(valid_result_output)

                    if self.rank == 0:
                        self.tensorboard.add_scalar(
                            "Valid_score", valid_score, epoch_idx
                        )
                        for pred_idx in self.metrics_pred_len_list:
                            for name, value in valid_result[f"pred_{pred_idx}"].items():
                                self.tensorboard.add_scalar(
                                    f"{name.replace('@', '_')}_pred_{pred_idx}",
                                    value,
                                    epoch_idx,
                                )
                            valid_result[f"pred_{pred_idx}"]["Pred Metrics"] = f"pred_{pred_idx}"
                            valid_result[f"pred_{pred_idx}"]["Train Steps"] = self.train_step
                            self.results_df.loc[len(self.results_df)] = valid_result[f"pred_{pred_idx}"]
                        self.results_df = self.results_df.sort_values(by=["Pred Metrics", "Train Steps"])
                        self.results_df.to_pickle(f"{self.saved_model_name[:-4]}_results.pkl")

                    if update_flag:
                        if saved:
                            self._save_checkpoint(self.train_step, verbose=verbose)
                        self.best_valid_result = valid_result

                    self.wandblogger.log_metrics(
                        {**{"score": self.best_valid_score}, **self.best_valid_result},
                        step=self.train_step,
                        head="best_valid",
                    )

                    if callback_fn:
                        callback_fn(epoch_idx, valid_score)

                    if stop_flag:
                        stop_output = f"Finished training, best eval result at step "\
                                      f"{self.train_step - self.no_improve_times * self.eval_interval}"
                        if verbose:
                            self.logger.info(stop_output)
                        break
                    torch.cuda.empty_cache()
                    gc.collect()

        if self.rank == 0:
            for pred_idx in self.metrics_pred_len_list:
                self.best_valid_result[f"pred_{pred_idx}"]["Train Steps"] = "Best Valid"
                self.results_df.loc[len(self.results_df)] = self.best_valid_result[f"pred_{pred_idx}"]
        self.model.eval()
        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def _full_sort_batch_eval(self, batched_data):
        _, item_seq, item_target, history_index, positive_u, time_seq, \
            target_tags, outlier_users = batched_data

        item_seq = self.to_device(item_seq, non_blocking=True)
        time_seq = self.to_device(time_seq, non_blocking=True)
        target_tags = self.to_device(target_tags, non_blocking=True)
        positive_u = self.to_device(positive_u, non_blocking=True)
        item_target = self.to_device(item_target, non_blocking=True)
        # self.item_feature is the embeddings of all items from the item_llm
        if self.config["model"] == "HLLM":
            if self.config["stage"] == 3:
                scores, wandb_logs, saved_user_embs, saved_head_embs = self.model.module.predict(
                    item_seq, time_seq, self.item_feature, self.all_item_tags, target_tags, save_for_eval=self.save_for_eval
                )
            else:
                scores, wandb_logs, saved_user_embs, saved_head_embs = self.model(
                    (item_seq, time_seq, self.item_feature, self.all_item_tags, target_tags, self.save_for_eval), mode="predict"
                )
        else:
            scores, wandb_logs, saved_user_embs, saved_head_embs = self.model.module.predict(
                item_seq, time_seq, self.item_feature, self.all_item_tags, target_tags, save_for_eval=self.save_for_eval
            )
        del item_seq, time_seq

        scores[:, :, 0] = -np.inf
        if not self.config["debug"] and self.config.get("suppress_history", True) and history_index is not None:
            scores[history_index[0], :, history_index[1]] = -np.inf
        del history_index

        return scores, positive_u, item_target, target_tags, outlier_users, wandb_logs, saved_user_embs, saved_head_embs

    @torch.no_grad()
    def compute_item_feature(self, config, data):
        self.model.eval()
        all_item_tags, all_original_item_tags = [], []
        if self.use_text:
            item_data = BatchTextDataset(config, data)
            item_batch_size = (
                config["MAX_ITEM_LIST_LENGTH"] * config["train_batch_size"]
            )
            item_loader = DataLoader(
                item_data,
                batch_size=item_batch_size,
                num_workers=6,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=4, 
                collate_fn=customize_rmpad_collate,
            )
            self.item_feature = []
            gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad():
                self.model.eval()
                iter_data = (
                    tqdm(
                        enumerate(item_loader),
                        total=len(item_loader),
                        miniters=self.update_interval,
                        bar_format="{l_bar}{bar}\n{r_bar}",
                        ncols=150,
                        desc=set_color(f"Item Features ", "pink"),
                        ascii=True,
                        file=sys.stdout,
                    )
                    if self.rank == 0
                    else enumerate(item_loader)
                )
                for idx, items in iter_data:
                    items = self.to_device(items, non_blocking=True)
                    items, item_tags, original_item_tags = self.model(items, mode="compute_item")
                    self.item_feature.append(items)
                    all_item_tags.append(item_tags)
                    all_original_item_tags.append(original_item_tags)
                del item_loader, item_data

                if isinstance(items, tuple):
                    self.item_feature = torch.cat(
                        [x[0] for x in self.item_feature]
                    ), torch.cat([x[1] for x in self.item_feature])
                else:
                    self.item_feature = torch.cat(self.item_feature)
                all_item_tags = torch.cat(all_item_tags)
                all_original_item_tags = torch.cat(all_original_item_tags)
                if self.config["stage"] == 3:
                    self.item_feature = self.item_feature.bfloat16()

        else:
            with torch.no_grad():
                self.model.eval()
                self.item_feature = self.model.module.compute_item_all()
                item_data = BatchItemDataset(config, data)
                item_batch_size = (
                    config["MAX_ITEM_LIST_LENGTH"] * config["train_batch_size"]
                )
                item_loader = DataLoader(
                    item_data,
                    batch_size=item_batch_size,
                    num_workers=4,
                    shuffle=False,
                    pin_memory=True,
                    collate_fn=customize_rmpad_collate,
                )
                gc.collect()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for items in item_loader:
                        items = self.to_device(items, non_blocking=True)
                        all_item_tags.append(items['pos_tag_categories'])
                        all_original_item_tags.append(items['pos_original_tag_categories'])
                    del item_loader
                    all_item_tags = torch.cat(all_item_tags)
                    all_original_item_tags = torch.cat(all_original_item_tags)
                del item_data
        self.model.train()
        self.logger.info(f'all_item_tags: {all_item_tags.shape}')
        if self.save_for_eval and self.rank == 0:
            os.makedirs(f"/workplace/saved_eval/{self.saved_model_name[:-4]}_eval_data", exist_ok=True)
            self.logger.info(f'Saving to /workplace/saved_eval/{self.saved_model_name[:-4]}_eval_data')
            torch.save(all_item_tags, f"/workplace/saved_eval/{self.saved_model_name[:-4]}_eval_data/all_item_tags.pt")
            torch.save(self.item_feature, f"/workplace/saved_eval/{self.saved_model_name[:-4]}_eval_data/item_feature.pt")
            save_strings(data.id2token['user_id'], f"/workplace/saved_eval/{self.saved_model_name[:-4]}_eval_data/user_id.txt.gz")
            save_strings(data.id2token['item_id'], f"/workplace/saved_eval/{self.saved_model_name[:-4]}_eval_data/item_id.txt.gz")
        self.eval_collector.set_all_tags(all_original_item_tags)
        self.all_item_tags = all_item_tags.transpose(0, 1)

    def evaluate(
        self,
        eval_data,
        load_best_model=True,
        show_progress=False,
        init_model=False,
    ):
        """
        1. Collector (eval_collector) inits DataStruct
        2. compute_item_feature(): Compute the embeddings of all candidate items, and save them as self.item_feature.
        Collect all item tags, and save them as self.all_item_tags.
        3. eval_func (self._full_sort_batch_eval): Given seq_len of interactions, each head outputs an embedding,
        with which we compute the score for all candidate items for each head  <<== Each head can only have score for the same category (need tag for all items in batchset)
        4. Collector.eval_batch_collect: how to combine the scores from different heads.
        Mark all target_items as 1s. Gather top-K recommended items from each head. Store as a tensor in DataStruct.
        5. Collector.get_data_struct retrieves the DataStruct for the desired N-target metrics, and parse it into
        Evaluator.evaluate().
        """

        if not eval_data:
            return
        if init_model:
            self.logger.info("set up fabric with model and optimizer")
            self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)

        if load_best_model:
            if "full_model_fp32.pt" in os.listdir(self.saved_model_file):
                shard_dir = os.path.join(self.saved_model_file, "full_model_fp32.pt")
                if os.path.isfile(os.path.join(shard_dir, "pytorch_model.bin")):
                    state = torch.load(os.path.join(shard_dir, "pytorch_model.bin"), map_location="cpu")
                else:
                    with open(os.path.join(shard_dir, "pytorch_model.bin.index.json")) as f:
                        index = json.load(f)
                    state = {}
                    for shard_file in tqdm(sorted(set(index["weight_map"].values()))):
                        state.update(torch.load(os.path.join(shard_dir, shard_file), map_location="cpu"))  # fp32

                missing, unexpected = self.model.load_state_dict(state, strict=False)
                self.logger.info(f"Missing keys: {missing}")
                self.logger.info(f"Unexpected keys: {unexpected}")
                if self.config['fix_temp']:
                    missing = set(missing) - {'logit_scale'}
                if missing:
                    self.logger.info(f"Missing keys: {missing}")
                    self.logger.info(f"Unexpected keys: {unexpected}")
                    raise ValueError(f"Missing keys: {missing}")

            else:
                state = {"model": self.model}
                self.lite.load(self.saved_model_file, state, strict=False)
            self.logger.info(f"Loading model structure and parameters from {self.saved_model_file}")

        with torch.no_grad():
            self.model.eval()
            eval_func = self._full_sort_batch_eval
            final_wandb_logs = defaultdict(float)

            self.tot_item_num = eval_data.dataset.dataload.item_num
            if not self.freeze_item_llm:
                self.logger.info('Start computing item features...')
                self.compute_item_feature(self.config, eval_data.dataset.dataload)
                self.logger.info("Finished computing item features...")
            iter_data = (
                tqdm(
                    eval_data,
                    bar_format="{l_bar}{bar}\n{r_bar}",
                    miniters=self.update_interval,
                    total=len(eval_data),
                    ncols=150,
                    desc=set_color(f"Evaluate   ", "pink"),
                    ascii=True,
                    file=sys.stdout,
                )
                if show_progress and self.rank == 0
                else eval_data
            )
            eval_time = t.time()
            log_dict = {}  # used only when self.log_detailed_results is True
            if self.save_for_eval:
                all_scores = []
                all_users = []
                all_item_history = []
                all_item_tgt = []
                all_user_embeddings = []
                all_head_embeddings = []
            save_for_eval_div = 0
            save_for_eval_rmd = 0
                
            for batch_idx, batched_data in enumerate(iter_data):
                start_time = eval_time
                data_time = t.time()
                log_this_batch = (self.rank == 0 and self.log_detailed_results)
                if log_this_batch:
                    user_np = batched_data[0].data.cpu().tolist()
                    log_dict["user"] = [
                        eval_data.dataset.dataload.id2token["user_id"][user_id]
                        for user_id in user_np
                    ]
                    item_history_np = batched_data[1].data.cpu().tolist()
                    log_dict["item_history"] = [
                        [eval_data.dataset.dataload.id2token["item_id"][item_id]  # front padded with zeros
                        for item_id in item_session]
                        for item_session in item_history_np
                    ]
                    item_tgt_np = batched_data[2].data.cpu().tolist()
                    log_dict["item_tgt"] = [
                        [eval_data.dataset.dataload.id2token["item_id"][item_id]
                        for item_id in item_session]
                        for item_session in item_tgt_np
                    ]

                scores, positive_u, positive_i, target_tags, outlier_users, wandb_logs, saved_user_embs, saved_head_embs = eval_func(batched_data)

                if self.save_for_eval:
                    all_scores.append(scores.float().cpu().numpy())
                    all_users.append(batched_data[0].data.cpu())
                    all_item_history.append(batched_data[1].float().data.cpu().numpy())
                    all_item_tgt.append(batched_data[2].float().data.cpu().numpy())
                    all_user_embeddings.append(saved_user_embs)
                    all_head_embeddings.append(saved_head_embs)
                    save_for_eval_div += 1
                    if save_for_eval_div >= 10:
                        save_for_eval_rmd += 1
                        all_scores = np.concatenate(all_scores, axis=0)
                        all_users = np.concatenate(all_users, axis=0)
                        all_item_history = np.concatenate(all_item_history, axis=0)
                        all_item_tgt = np.concatenate(all_item_tgt, axis=0)
                        if save_for_eval_rmd < 2:
                            all_user_embeddings = np.concatenate(all_user_embeddings, axis=0)
                            all_head_embeddings = np.concatenate(all_head_embeddings, axis=0)
                        save_for_eval_div = 0
                        if save_for_eval_rmd < 2:
                            np.savez(f"/workplace/saved_eval/{self.saved_model_name[:-4]}_eval_data/eval_data_{save_for_eval_rmd}_rank{self.rank}.npz", scores=all_scores, users=all_users, item_history=all_item_history, item_tgt=all_item_tgt, user_embeddings=all_user_embeddings, head_embeddings=all_head_embeddings)
                        else:
                            np.savez(f"/workplace/saved_eval/{self.saved_model_name[:-4]}_eval_data/eval_data_{save_for_eval_rmd}_rank{self.rank}.npz", scores=all_scores, users=all_users, item_history=all_item_history, item_tgt=all_item_tgt)
                        all_scores = []
                        all_users = []
                        all_item_history = []
                        all_item_tgt = []
                        all_user_embeddings = []
                        all_head_embeddings = []
                # scores: torch.Size([512, 12, 398261]), positive_u: torch.Size([512, 8]), positive_i: torch.Size([512, 8])
                # target_tags: torch.Size([512, 8, 8]), outlier_users: torch.Size([512])
                del batched_data
                if self.rank == 0 and self.prior_switch is not None:
                    if self.config.get('master_switch', False):
                        prior_switch_num_cats = 1
                    else:
                        prior_switch_num_cats = self.eval_num_cats
                    for head_idx in range(prior_switch_num_cats):
                        prefix = f'head_cat_{self.int_to_category[head_idx]}_'
                        current_num_correct = wandb_logs[f'{prefix}num_correct']
                        final_wandb_logs[f'{prefix}num_correct'] += current_num_correct.item()
                        self.logger.info(f'current_num_correct: {current_num_correct.item()}, final: {final_wandb_logs[f"{prefix}num_correct"]}')
                    current_num_samples = wandb_logs['num_samples']
                    final_wandb_logs['num_samples'] += current_num_samples
                    self.logger.info(f'current_num_samples: {current_num_samples}, final: {final_wandb_logs["num_samples"]}')
                fwd_time = t.time()

                top_scores_by_head = self.eval_collector.eval_batch_collect(
                    scores, positive_u, positive_i,
                    tag_category=target_tags if self.eval_by_cat else None,
                    outlier_users=outlier_users if self.outlier_user_metrics is not None else None,
                    log_detailed_results=log_this_batch
                )

                eval_time = t.time()
                if show_progress and self.rank == 0:
                    iter_data.set_postfix_str(
                        f"data: {data_time - start_time:.3f} fwd: {fwd_time - data_time:.3f} eval: {eval_time - fwd_time:.3f}",
                        refresh=False,
                    )

                if log_this_batch:
                    top_scores_by_head['recommend_items'] = [
                        [eval_data.dataset.dataload.id2token["item_id"][item_id]  # front padded with zeros
                         for item_id in item_session]
                        for item_session in top_scores_by_head['idx']
                    ]
                    del top_scores_by_head['idx']
                    if 'idx_by_head' in top_scores_by_head:
                        top_scores_by_head['recommend_items_by_head'] = [[
                            [eval_data.dataset.dataload.id2token["item_id"][item_id]
                             for item_id in session_head] for session_head in item_session]
                            for item_session in top_scores_by_head['idx_by_head']
                        ]
                        del top_scores_by_head['idx_by_head']
                    log_dict.update(top_scores_by_head)
                    save_log_dict(log_dict, prefix=f"step{self.train_step}-idx{batch_idx}-",
                        folder_path=self.saved_model_file[:-4])

            del iter_data
            if self.rank == 0:
                log_wandb_dict = dict()
                if self.prior_switch is not None:
                    if self.config.get('master_switch', False):
                        prior_switch_num_cats = 1
                    else:
                        prior_switch_num_cats = self.eval_num_cats
                    for head_idx in range(prior_switch_num_cats):
                        prefix = f'head_cat_{self.int_to_category[head_idx]}_'
                        log_wandb_dict[f'{prefix}acc'] = \
                            final_wandb_logs[f'{prefix}num_correct'] / final_wandb_logs['num_samples']

                if len(log_wandb_dict) > 0:
                    self.logger.info(f'=========> {log_wandb_dict}')
                    log_head = 'test' if load_best_model else 'valid'
                    self.wandblogger.log_metrics(
                        log_wandb_dict, step=self.train_step, head=log_head
                    )
            torch.distributed.barrier()

            if self.log_detailed_results:
                num_total_examples = len(eval_data.dataset)
            else:
                num_total_examples = len(eval_data.sampler.dataset)
            result_summary = dict()
            metric_decimal_place = \
                5 if self.config["metric_decimal_place"] is None else self.config["metric_decimal_place"]

            struct = self.eval_collector.get_data_struct(-1)  # attributes shared across pred_len
            if struct is not None:
                result = self.evaluator.evaluate(struct, pred_len=-1)
                del struct
                gc.collect()
                torch.cuda.empty_cache()
                result_keys = sorted(list(result.keys()))
                for k in result_keys:
                    if isinstance(result[k], tuple):
                        v, num_selected_examples = result[k]
                    else:
                        v = result[k]
                        num_selected_examples = num_total_examples
                    torch.distributed.barrier()
                    tensor_val = torch.tensor([v], dtype=torch.float32).to(self.device)
                    torch.distributed.all_reduce(tensor_val, op=dist.ReduceOp.SUM)
                    if isinstance(result[k], tuple):
                        tensor_num = torch.tensor(
                            [num_selected_examples], dtype=torch.float32
                        ).to(self.device)
                        torch.distributed.all_reduce(tensor_num, op=dist.ReduceOp.SUM)
                        result[k] = round(
                            tensor_val.item() / max(1, tensor_num.item()), metric_decimal_place
                        )
                    else:
                        result[k] = round(
                            tensor_val.item() / max(1, num_selected_examples),
                            metric_decimal_place,
                        )

                    del tensor_val
                    gc.collect()
                    torch.cuda.empty_cache()

                if load_best_model and self.rank == 0:  # only testing loads best model
                    self.wandblogger.log_metrics(
                        result, step=self.train_step, head=f'test_shared'
                    )

                result_summary['shared'] = result

            self.eval_collector.reset_all_tags()

            for pred_idx in self.metrics_pred_len_list:
                struct = self.eval_collector.get_data_struct(pred_idx)
                result = self.evaluator.evaluate(struct, pred_len=pred_idx)
                del struct
                gc.collect()
                torch.cuda.empty_cache()

                result_keys = sorted(list(result.keys()))
                local_metrics = {}

                for k in result_keys:
                    if isinstance(result[k], tuple):
                        v, num_selected_examples = result[k]
                    else:
                        v = result[k]
                        num_selected_examples = num_total_examples
                    local_metrics[k] = (v, num_selected_examples, isinstance(result[k], tuple))

                torch.distributed.barrier()

                for k, (v, num_selected_examples, is_tuple) in local_metrics.items():
                    tensor_val = torch.tensor([v], dtype=torch.float32).to(self.device)
                    torch.distributed.all_reduce(tensor_val, op=dist.ReduceOp.SUM)

                    if is_tuple:
                        tensor_num = torch.tensor([num_selected_examples], dtype=torch.float32).to(self.device)
                        torch.distributed.all_reduce(tensor_num, op=dist.ReduceOp.SUM)
                        result[k] = round(
                            tensor_val.item() / max(1, tensor_num.item()), metric_decimal_place
                        )
                    else:
                        result[k] = round(
                            tensor_val.item() / max(1, num_selected_examples),
                            metric_decimal_place,
                        )

                    del tensor_val
                    if is_tuple:
                        del tensor_num

                gc.collect()
                torch.cuda.empty_cache()

                if load_best_model and self.rank == 0:  # only testing loads best model
                    self.wandblogger.log_metrics(
                        result, step=self.train_step, head=f"test_pred_{pred_idx}"
                    )

                result_summary[f"pred_{pred_idx}"] = result

            torch.distributed.barrier()

            if load_best_model and self.rank == 0:  # Test only
                results_summary_save = copy.deepcopy(result_summary)
                for pred_idx in self.metrics_pred_len_list:
                    if 'shared' in results_summary_save:
                        results_summary_save['shared']["Train Steps"] = "Test"
                        self.results_df.loc[len(self.results_df)] = results_summary_save['shared']
                    results_summary_save[f"pred_{pred_idx}"]["Train Steps"] = "Test"
                    self.results_df.loc[len(self.results_df)] = results_summary_save[
                        f"pred_{pred_idx}"
                    ]
                self.results_df.to_pickle(f"{self.saved_model_name[:-4]}_results.pkl")

        return result_summary