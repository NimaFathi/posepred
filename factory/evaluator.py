import logging
import time

import torch
import numpy as np
import os
from tqdm import tqdm

from metrics import POSE_METRICS, MASK_METRICS
from utils.others import dict_to_device

logger = logging.getLogger(__name__)


class Evaluator:
    # evaluator = Evaluator(cfg, eval_dataloader, model, loss_module, eval_reporter)
    def __init__(self, args, dataloader, model, loss_module, reporter):
        self.args = args
        self.dataloader = dataloader
        self.model = model.to(args.device)
        self.loss_module = loss_module.to(args.device)
        self.reporter = reporter
        self.is_interactive = args.data.is_interactive
        self.pose_metrics = args.pose_metrics
        self.mask_metrics = args.mask_metrics
        self.rounds_num = args.rounds_num
        self.device = args.device

    def evaluate(self):
        logger.info('Evaluation started.')
        self.model.eval()
        # self.loss_module.eval()
        for i in range(self.rounds_num):
            logger.info('round ' + str(i + 1) + '/' + str(self.rounds_num))
            self.__evaluate()
        
        pose_metrics_name = [f'{metric_name}{tm}ms' for metric_name in self.pose_metrics for tm in [80, 160, 320, 400, 560, 720, 880, 1000]]

        self.reporter.print_pretty_metrics(logger, self.model.args.use_mask, pose_metrics_name)
        self.reporter.save_csv_metrics(self.model.args.use_mask, pose_metrics_name, os.path.join(self.args.csv_save_dir,"eval.csv"))
        logger.info("Evaluation has been completed.")

    def __evaluate(self):
        self.reporter.start_time = time.time()
        pose_key = None
        for data in tqdm(self.dataloader):
            actions = set(data['action']) if 'action' in data.keys() else set()
            actions.add("all")
            # TODO
            if pose_key is None:
                pose_key = [k for k in data.keys() if "pose" in k][0]
            batch_size = data[pose_key].shape[0]
            with torch.no_grad():
                # predict & calculate loss
                model_outputs = self.model(dict_to_device(data, self.device))
                loss_outputs = self.loss_module(model_outputs, dict_to_device(data, self.device))
                assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'

                if self.model.args.use_mask:
                    assert 'pred_mask' in model_outputs.keys(), 'outputs of model should include pred_mask'
                    pred_mask = model_outputs['pred_mask']
                else:
                    pred_mask = None

                # calculate pose_metrics
                report_attrs = loss_outputs
                dynamic_counts = {}
                for metric_name in self.pose_metrics:
                    metric_func = POSE_METRICS[metric_name]

                    pred_metric_pose = model_outputs['pred_pose']
                    if 'pred_metric_pose' in model_outputs:
                        pred_metric_pose = model_outputs['pred_metric_pose']

                    # TODO: write write a warning =D

                    future_metric_pose = data['future_pose']
                    if 'future_metric_pose' in data:
                        future_metric_pose = data['future_metric_pose']

                    for tm in [80, 160, 320, 400, 560, 720, 880, 1000]:
                        fm = 25 * tm // 1000

                        for action in actions:
                            if action == "all":
                                metric_value = metric_func(pred_metric_pose.to(self.device)[:, :fm],
                                                        future_metric_pose.to(self.device)[:, :fm],
                                                        self.model.args.keypoint_dim, pred_mask)
                            else:
                                indexes = np.where(np.asarray(data['action']) == action)[0]
                                metric_value = metric_func(pred_metric_pose.to(self.device)[indexes][:, :fm],
                                                        future_metric_pose.to(self.device)[indexes][:, :fm],
                                                        self.model.args.keypoint_dim, pred_mask)
                                dynamic_counts[f'{metric_name}{tm}ms_{action}']=len(indexes)
                            report_attrs[f'{metric_name}{tm}ms_{action}'] = metric_value

                # calculate mask_metrics
                if self.model.args.use_mask:
                    for metric_name in self.mask_metrics:
                        metric_func = MASK_METRICS[metric_name]
                        metric_value = metric_func(pred_mask, data['future_mask'].to(self.device), self.device)
                        report_attrs[metric_name] = metric_value

                self.reporter.update(report_attrs, batch_size, True, dynamic_counts)

        self.reporter.epoch_finished()
