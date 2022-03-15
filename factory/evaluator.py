import logging
import time

import torch

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
        for i in range(self.rounds_num):
            logger.info('round ' + str(i + 1) + '/' + str(self.rounds_num))
            self.__evaluate()
            #for metric in self.args.pose_metrics:
            #    logger.info(f'{metric}: ' + str(self.reporter.history[metric][-1]))
            # logger.info(f'MSE: ' + str(self.reporter.history['MSE'][-1]))
        self.reporter.print_pretty_metrics(logger, self.model.args.use_mask, self.pose_metrics)
        self.reporter.save_csv_metrics(self.model.args.use_mask, self.pose_metrics, None)
        logger.info("Evaluation has been completed.")

    def __evaluate(self):
        self.reporter.start_time = time.time()
        pose_key = None
        for data in self.dataloader:
            actions = set(data['action']) if 'action' in data.keys() else set()
            actions.add("all")
            if pose_key is None:
                pose_key = [k for k in data.keys() if "pose" in k][0]
            batch_size =data[pose_key].shape[0]
            with torch.no_grad():
                # predict & calculate loss
                model_outputs = self.model(dict_to_device(data, self.device))
                loss_outputs = self.loss_module(model_outputs, dict_to_device(data, self.device))
                model_pose_format = "_"+self.args.model_pose_format if self.args.model_pose_format!= "" else ""
                metric_pose_format = "_"+self.args.metric_pose_format if self.args.metric_pose_format!= "" else ""
                assert f'pred{model_pose_format}_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'

                if self.model.args.use_mask:
                    assert 'pred_mask' in model_outputs.keys(), 'outputs of model should include pred_mask'
                    pred_mask = model_outputs['pred_mask']
                else:
                    pred_mask = None

                # calculate pose_metrics
                report_attrs = loss_outputs
                for metric_name in self.pose_metrics:
                    metric_func = POSE_METRICS[metric_name]
                    
                    for action in actions:
                        if action == "all":
                            metric_value = metric_func(model_outputs[f'pred{model_pose_format}_pose'].to(self.device), data[f'future{metric_pose_format}_pose'].to(self.device),
                                                        self.model.args.pred_keypoint_dim, pred_mask)
                            report_attrs[f'{metric_name}_all'] = metric_value
                        else:
                            indexes = [i==action for i in data['action']]
                            metric_value = metric_func(model_outputs[f'pred{model_pose_format}_pose'].to(self.device)[indexes], data[f'future{metric_pose_format}_pose'].to(self.device)[indexes],
                                                        self.model.args.pred_keypoint_dim, pred_mask)
                            report_attrs[f'{metric_name}_{action}'] = metric_value
                # print(report_attrs)
                # calculate mask_metrics
                if self.model.args.use_mask:
                    for metric_name in self.mask_metrics:
                        metric_func = MASK_METRICS[metric_name]
                        metric_value = metric_func(pred_mask, data['future_mask'].to(self.device), self.device)
                        report_attrs[metric_name] = metric_value

                self.reporter.update(report_attrs, batch_size, True)

        self.reporter.epoch_finished()
