import logging
import time

import torch
import numpy as np
import os
from tqdm import tqdm

from metrics import POSE_METRICS, MASK_METRICS
from utils.others import dict_to_device
from uncertainty.utils.uncertainty import calculate_dict_uncertainty, UNC_K
from uncertainty.utils.prediction_util import get_prediction_model_dict

logger = logging.getLogger(__name__)


class UncertaintyEvaluator:
    def __init__(self, args, dataloader, model, uncertainty_model, input_n, output_n, dataset_name, reporter):
        self.args = args
        self.dataloader = dataloader
        self.model = model.to(args.device)
        self.uncertainty_model = uncertainty_model.to(args.device)
        self.dataset_name = dataset_name
        self.batch_size = args.data.batch_size
        self.reporter = reporter
        self.device = args.device

    def evaluate(self):
        logger.info('Uncertainty evaluation started.')
        self.model.eval()
        self.__evaluate()
        self.reporter.print_pretty_uncertainty(logger, UNC_K)
        # self.reporter.save_csv_metrics(self.model.args.use_mask, self.pose_metrics,
        #                                os.path.join(self.args.csv_save_dir, "eval.csv"))
        logger.info("Uncertainty evaluation has been completed.")

    def __evaluate(self):
        self.reporter.start_time = time.time()
        model_dict = get_prediction_model_dict(self.model, self.dataloader, self.device)
        self.uncertainty = calculate_dict_uncertainty(self.dataset_name, model_dict, self.uncertainty_model,
                                                      self.batch_size, self.device)
        self.reporter.update(self.uncertainty, 1, False)
        self.reporter.epoch_finished()
