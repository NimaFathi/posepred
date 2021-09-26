import logging
import time
from logging import config

import torch

from path_definition import LOGGER_CONF
from utils.losses import L1, MSE, BCE
from utils.metrics import accuracy, ADE, FDE
from utils.others import pose_from_vel

config.fileConfig(LOGGER_CONF)
logger = logging.getLogger('evalLogger')


class Evaluator:
    def __init__(self, model, dataloader, reporter, is_interactive, distance_loss, rounds_num):
        self.model = model
        self.dataloader = dataloader
        self.reporter = reporter
        self.is_interactive = is_interactive
        self.distance_loss = L1() if distance_loss == 'L1' else MSE()
        self.rounds_num = rounds_num
        self.mask_loss = BCE()
        self.device = torch.device('cuda')

    def evaluate(self):
        self.model.eval()
        logger.info('Evaluation started ...')
        for i in range(self.rounds_num):
            logger.info('round', i + 1)
            self.__evaluate()
        logger.info('-' * 100)
        self.reporter.print_mean_std(logger, self.model.args.use_mask)
        logger.info("Evaluation has been completed")

    def __evaluate(self):
        self.reporter.start_time = time.time()
        for data in self.dataloader:
            for i, d in enumerate(data):
                data[i] = d.to(self.device)
            batch_size = data[0].shape[0]
            if self.model.args.use_mask:
                obs_pose, obs_vel, obs_mask, target_pose, target_vel, target_mask = data
            else:
                obs_pose, obs_vel, target_pose, target_vel = data

            with torch.no_grad():
                # predict
                outputs = self.model(data[:len(data) // 2])

                # calculate metrics
                pred_vel = outputs[0]
                vel_loss = self.distance_loss(pred_vel, target_vel)
                pred_pose = pose_from_vel(pred_vel, obs_pose[..., -1, :])
                report_metrics = {'vel_loss': vel_loss}
                if self.model.args.use_mask:
                    pred_mask = outputs[1]
                    mask_loss = self.mask_loss(pred_mask, target_mask)
                    mask_acc = accuracy(pred_mask, target_mask)
                    report_metrics['mask_loss'] = mask_loss
                    report_metrics['mask_acc'] = mask_acc
                ade = ADE(pred_pose, target_pose, self.model.args.keypoint_dim)
                fde = FDE(pred_pose, target_pose, self.model.args.keypoint_dim)
                report_metrics['ADE'] = ade
                report_metrics['FDE'] = fde
                self.reporter.update(report_metrics, batch_size)

        self.reporter.epoch_finished()
