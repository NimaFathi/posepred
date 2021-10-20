import time
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from metrics import POSE_METRICS, MASK_METRICS
from utils.reporter import Reporter
from utils.save_load import save_snapshot
from utils.others import dict_to_device

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, train_dataloader, valid_dataloader, model, loss_module, optimizer, optimizer_args,
                 scheduler, train_reporter, valid_reporter):
        self.args = args
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model.to(args.device)
        self.loss_module = loss_module.to(args.device)
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.scheduler = scheduler
        self.train_reporter = train_reporter
        self.valid_reporter = valid_reporter
        self.tensor_board = SummaryWriter(args.save_dir)
        self.use_validation = False if valid_dataloader is None else True

    def train(self):
        logger.info("Training started.")
        time0 = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.__train()
            if self.use_validation:
                self.__validate()
                self.scheduler.step(self.valid_reporter.history['loss'][-1])
            if (epoch + 1) % self.args.snapshot_interval == 0 or (epoch + 1) == self.args.epochs:
                save_snapshot(self.model, self.loss_module, self.optimizer, self.optimizer_args, epoch + 1,
                              self.train_reporter,
                              self.valid_reporter, self.args.save_dir)
                self.train_reporter.save_data(self.model.args.use_mask, self.args.save_dir)
                if self.use_validation:
                    self.valid_reporter.save_data(self.model.args.use_mask, self.args.save_dir)
                Reporter.save_plots(self.model.args.use_mask, self.args.save_dir, self.train_reporter.history,
                                    self.valid_reporter.history, self.use_validation)
        self.tensor_board.close()
        logger.info("-" * 100)
        logger.info('Training is completed in %.2f seconds.' % (time.time() - time0))

    def __train(self):
        self.model.train()
        self.train_reporter.start_time = time.time()
        for data in self.train_dataloader:
            batch_size = data['observed_pose'].shape[0]
            data = dict_to_device(data, self.args.device)
            # predict & calculate loss
            self.model.zero_grad()
            model_outputs = self.model(data)
            loss_outputs = self.loss_module(model_outputs, dict_to_device(data, self.args.device))
            assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'
            assert 'loss' in loss_outputs.keys(), 'outputs of loss should include loss'

            # backpropagate and optimize
            loss = loss_outputs['loss']
            loss.backward()
            self.optimizer.step()

            if self.model.args.use_mask:
                assert 'pred_mask' in model_outputs.keys(), 'outputs of model should include pred_mask'
                pred_mask = model_outputs['pred_mask']
            else:
                pred_mask = None

            # calculate pose_metrics
            report_attrs = loss_outputs
            for metric_name in self.args.pose_metrics:
                metric_func = POSE_METRICS[metric_name]
                metric_value = metric_func(model_outputs['pred_pose'], data['future_pose'].to(self.args.device),
                                           self.model.args.keypoint_dim, pred_mask)
                report_attrs[metric_name] = metric_value

            # calculate mask_metrics
            if self.model.args.use_mask:
                for metric_name in self.args.mask_metrics:
                    metric_func = MASK_METRICS[metric_name]
                    metric_value = metric_func(pred_mask, data['future_mask'].to(self.args.device), self.args.device)
                    report_attrs[metric_name] = metric_value

            self.train_reporter.update(report_attrs, batch_size)

        self.train_reporter.epoch_finished(self.tensor_board)
        self.train_reporter.print_values(logger, self.model.args.use_mask)

    def __validate(self):
        self.model.eval()
        self.valid_reporter.start_time = time.time()
        for data in self.valid_dataloader:
            data = dict_to_device(data, self.args.device)
            batch_size = data['observed_pose'].shape[0]

            with torch.no_grad():
                # predict & calculate loss
                model_outputs = self.model(data)
                loss_outputs = self.loss_module(model_outputs, dict_to_device(data, self.args.device))
                assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'

                if self.model.args.use_mask:
                    assert 'pred_mask' in model_outputs.keys(), 'outputs of model should include pred_mask'
                    pred_mask = model_outputs['pred_mask']
                else:
                    pred_mask = None

                # calculate pose_metrics
                report_attrs = loss_outputs
                for metric_name in self.args.pose_metrics:
                    metric_func = POSE_METRICS[metric_name]
                    metric_value = metric_func(model_outputs['pred_pose'], data['future_pose'],
                                               self.model.args.keypoint_dim, pred_mask)
                    report_attrs[metric_name] = metric_value

                # calculate mask_metrics
                if self.model.args.use_mask:
                    for metric_name in self.args.mask_metrics:
                        metric_func = MASK_METRICS[metric_name]
                        metric_value = metric_func(pred_mask, data['future_mask'], self.args.device)
                        report_attrs[metric_name] = metric_value

                self.valid_reporter.update(report_attrs, batch_size)

        self.valid_reporter.epoch_finished(self.tensor_board)
        self.valid_reporter.print_values(logger, self.model.args.use_mask)
