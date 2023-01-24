import logging
import time

import torch
import numpy as np
import os
from tqdm import tqdm

from metrics import POSE_METRICS, MASK_METRICS
from utils.others import dict_to_device
from uncertainty.utils.uncertainty import calculate_dict_uncertainty_and_mpjpe
from uncertainty.utils.prediction_util import get_prediction_model_dict

logger = logging.getLogger(__name__)


class UncertaintyEvaluator:
    # evaluator = Evaluator(cfg, eval_dataloader, model, loss_module, eval_reporter)
    def __init__(self, args, dataloader, model, uncertainty_model, input_n, output_n, batch_size, dataset_name,
                 reporter):
        self.args = args
        self.dataloader = dataloader
        self.model = model.to(args.device)
        self.uncertainty_model = uncertainty_model.to(args.device)
        self.input_n = input_n
        self.batch_size = batch_size
        self.output_n = output_n
        self.dataset_name = dataset_name
        self.reporter = reporter
        self.is_interactive = args.data.is_interactive
        self.pose_metrics = args.pose_metrics
        self.mask_metrics = args.mask_metrics
        self.rounds_num = args.rounds_num
        self.device = args.device

    def evaluate(self):
        logger.info('Uncertainty evaluation started.')
        self.model.eval()
        # self.loss_module.eval()
        # for i in range(self.rounds_num):
        #     logger.info('round ' + str(i + 1) + '/' + str(self.rounds_num))
        self.__evaluate()
        self.reporter.print_pretty_metrics(logger, self.model.args.use_mask, self.pose_metrics)
        self.reporter.save_csv_metrics(self.model.args.use_mask, self.pose_metrics,
                                       os.path.join(self.args.csv_save_dir, "eval.csv"))
        logger.info("Uncertainty evaluation has been completed.")

    def __evaluate(self):
        self.reporter.start_time = time.time()
        model_dict = get_prediction_model_dict(self.model, self.dataloader, self.input_n, self.output_n,
                                               self.dataset_name, self.device)
        uncertainty_dict = calculate_dict_uncertainty_and_mpjpe(self.dataset_name, model_dict, self.uncertainty_model,
                                                                self.batch_size, self.device)
        self.reporter.update(uncertainty_dict, self.batch_size, True, 0)
        self.reporter.epoch_finished()
