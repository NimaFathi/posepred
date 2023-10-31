import logging
import time

import torch
import numpy as np
import os
from tqdm import tqdm

from utils.others import dict_to_device
from ..uncertainty.utils.uncertainty import calculate_dict_uncertainty_and_mpjpe, UNC_K, LOSS_K
from ..uncertainty.utils.prediction_util import get_prediction_model_dict

logger = logging.getLogger(__name__)


class UncertaintyEvaluator:
    def __init__(self, args, dataloader, model, uncertainty_model, input_n, output_n, dataset_name, reporter, in_line=False, save_dir='/home/posepred/final_yashar/posepred/'):
        self.args = args
        self.dataloader = dataloader
        self.model = model.to(args.device)
        self.uncertainty_model = uncertainty_model.to(args.device)
        self.dataset_name = dataset_name
        self.batch_size = args.data.batch_size
        self.reporter = reporter
        self.device = args.device
        self.save_dir = save_dir
        self.in_line = in_line
        self.input_n = input_n
        self.output_n = output_n

    def evaluate(self):
        self.model.eval()
        self.__evaluate()
        if self.in_line:
            self.reporter.print_uncertainty_values(logger, UNC_K)
            self.reporter.save_uncertainty_data(UNC_K, self.save_dir)
        else:
            logger.info('Uncertainty evaluation started.')
            self.reporter.print_pretty_uncertainty(logger, UNC_K)
            self.reporter.save_csv_uncertainty(UNC_K, os.path.join(self.args.csv_save_dir, "uncertainty_eval.csv"))
            logger.info("Uncertainty evaluation has been completed.")

    def __evaluate(self):
        self.reporter.start_time = time.time()
        model_dict = get_prediction_model_dict(self.model, self.dataloader, self.input_n, self.output_n,
                                               self.dataset_name, dev=self.device)
        self.result = calculate_dict_uncertainty_and_mpjpe(self.dataset_name, model_dict, self.uncertainty_model,
                                                      self.batch_size, self.device)
        self.uncertainty, self.mpjpe = self.result[UNC_K], self.result[LOSS_K]
        logger.info(f'## Uncertainty: {self.uncertainty}')
        self.reporter.update(self.uncertainty, 1, True, {UNC_K: 1})
        self.reporter.epoch_finished()
