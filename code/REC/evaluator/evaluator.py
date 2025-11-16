# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT
import torch

from .register import metrics_dict
from .collector import DataStruct
from collections import OrderedDict


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics.
    """

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config['metrics']]
        self.shared_metrics = [metric.lower() for metric in self.config['shared_metrics']]
        self.metric_class = {}
        for metric in self.metrics + self.shared_metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct, pred_len=1):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): Contains all the information needed for metrics for this pred_len.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        result_dict = OrderedDict()
        if pred_len == -1:
            metrics_list = self.shared_metrics
        else:
            metrics_list = self.metrics
        for metric in metrics_list:
            metric_val = self.metric_class[metric].calculate_metric(dataobject, pred_len=pred_len)
            result_dict.update(metric_val)
        return result_dict
