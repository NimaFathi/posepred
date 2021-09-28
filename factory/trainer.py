import time
import logging
from logging import config

import torch
from torch.utils.tensorboard import SummaryWriter

from path_definition import LOGGER_CONF
from losses import LOSSES
from metrics import POSE_METRICS, MASK_METRICS
from utils.others import pose_from_vel
from utils.reporter import Reporter
from utils.save_load import save_snapshot

config.fileConfig(LOGGER_CONF)
logger = logging.getLogger('trainLogger')


class Trainer:
    def __init__(self, trainer_args, model, train_dataloader, valid_dataloader, optimizer, scheduler, train_reporter,
                 valid_reporter):
        self.args = trainer_args
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_reporter = train_reporter
        self.valid_reporter = valid_reporter
        self.tb = SummaryWriter(trainer_args.save_dir)
        self.use_validation = False if valid_dataloader is None else True
        self.loss_module = LOSSES[self.args.loss_name]
        self.device = torch.device('cuda')

    def train(self):
        logger.info("Training started ...")
        self.model.train()
        time0 = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.__train()
            if self.use_validation:
                self.__validate()
                self.scheduler.step(self.valid_reporter.history['vel_loss'][-1])
            if (epoch + 1) % self.args.snapshot_interval == 0 or (epoch + 1) == self.args.epochs:
                save_snapshot(self.model, self.optimizer, self.args.lr, epoch + 1, self.train_reporter,
                              self.valid_reporter, self.args.save_dir)
                self.train_reporter.save_data(self.model.args.use_mask, self.args.save_dir)
                if self.use_validation:
                    self.valid_reporter.save_data(self.model.args.use_mask, self.args.save_dir)
                Reporter.save_plots(self.model.args.use_mask, self.args.save_dir, self.train_reporter.history,
                                    self.valid_reporter.history, self.use_validation)
        self.tb.close()
        logger.info("-" * 100)
        logger.info('Training is completed in %.2f seconds.' % (time.time() - time0))

    def __train(self):
        self.train_reporter.start_time = time.time()
        for data in self.train_dataloader:
            for key, value in data.items():
                data[key] = value.to(self.device)
            batch_size = data['observed_pose'].shape[0]

            # predict
            self.model.zero_grad()
            model_outputs = self.model(data)
            assert 'pred_pose' in model_outputs.keys(), 'model_outputs should include pred_pose'

            # calculate loss
            loss = self.loss_module(model_outputs, data)

            # calculate metrics

            if self.model.args.use_mask:
                assert 'pred_mask' in model_outputs.keys(), 'model_outputs should include pred_mask'
                # mask_acc = accuracy(pred_mask, target_mask)
                # report_metrics['mask_acc'] = mask_acc

            # report_metrics = {'vel_loss': vel_loss}
            # ade = ADE(pred_pose, target_pose, self.model.args.keypoint_dim)
            # fde = FDE(pred_pose, target_pose, self.model.args.keypoint_dim)
            # report_metrics['ADE'] = ade
            # report_metrics['FDE'] = fde
            self.train_reporter.update(report_metrics, batch_size)

            # backpropagate and optimize
            loss.backward()
            self.optimizer.step()

        self.train_reporter.epoch_finished(self.tb)
        self.train_reporter.print_values(logger, self.model.args.use_mask)

    def __validate(self):
        self.valid_reporter.start_time = time.time()
        for data in self.valid_dataloader:
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
                self.valid_reporter.update(report_metrics, batch_size)

        self.valid_reporter.epoch_finished(self.tb)
        self.valid_reporter.print_values(logger, self.model.args.use_mask)
