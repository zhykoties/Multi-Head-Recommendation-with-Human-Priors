# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT
from datetime import datetime
import pytz


def name_with_datetime():
    now = datetime.now(tz=pytz.utc)
    now = now.astimezone(pytz.timezone('US/Pacific'))
    return now.strftime("%Y-%m-%d_%H:%M")


class WandbLogger(object):
    """WandbLogger to log metrics to Weights and Biases.

    """

    def __init__(self, config, run_name):
        """
        Args:
            config (dict): A dictionary of parameters used by RecBole.
        """
        self.config = config
        self.log_wandb = config.log_wandb
        self.setup(run_name)

    def setup(self, run_name):
        if self.log_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                raise ImportError(
                    "To use the Weights and Biases Logger please install wandb."
                    "Run `pip install wandb` to install it."
                )

            # Initialize a W&B run
            if self._wandb.run is None:
                self._wandb.init(
                    entity="zhykoties",
                    project=self.config.wandb_project,
                    group=run_name,
                    config=self.config.recorded_config
                )

    def log_metrics(self, metrics, step, head='train'):
        if self.log_wandb:
            if head:
                metrics = self._add_head_to_metrics(metrics, head)
                self._wandb.log(metrics, step=step)
            else:
                self._wandb.log(metrics, step=step)

    def _add_head_to_metrics(self, metrics, head):
        head_metrics = dict()
        for k, v in metrics.items():
            head_metrics[f'{head}/{k}'] = v
        return head_metrics
