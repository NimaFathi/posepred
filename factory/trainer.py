import time
import torch

from utils.save_load import save_snapshot
from utils.metrics import accuracy, ADE, FDE
from utils.others import pose_from_vel
from utils.losses import L1, MSE, BCE
from utils.reporter import Reporter


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
        self.distance_loss = L1() if self.args.distance_loss == 'L1' else MSE()
        self.mask_loss = BCE()
        self.device = torch.device('cuda')

    def train(self):
        self.model.train()
        time0 = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.__train()
            self.__validate()
            self.scheduler.step(self.valid_reporter.history['vel_loss'][-1])
            if (epoch + 1) % self.args.snapshot_interval == 0 or (epoch + 1) == self.args.epochs:
                save_snapshot(self.model, self.optimizer, self.args.lr, epoch + 1, self.train_reporter,
                              self.valid_reporter, self.args.save_dir)
                self.train_reporter.save_data(self.model.args.use_mask, self.args.save_dir)
                self.valid_reporter.save_data(self.model.args.use_mask, self.args.save_dir)
                Reporter.save_plots(self.model.args.use_mask, self.args.save_dir, self.train_reporter.history,
                                    self.valid_reporter.history)
        print("-" * 100)
        print('Training is completed in %.2f seconds.' % (time.time() - time0))

    def __train(self):
        self.train_reporter.start_time = time.time()
        for data in self.train_dataloader:
            for i, d in enumerate(data):
                data[i] = d.to(self.device)
            batch_size = data[0].shape[0]

            if self.model.args.use_mask:
                obs_pose, obs_vel, obs_mask, target_pose, target_vel, target_mask = data
            else:
                obs_pose, obs_vel, target_pose, target_vel = data

            # predict
            self.model.zero_grad()
            outputs = self.model(data[:len(data) // 2])

            # calculate metrics
            pred_vel = outputs[0]
            vel_loss = self.distance_loss(pred_vel, target_vel)
            pred_pose = pose_from_vel(pred_vel, obs_pose[..., -1, :])
            loss = vel_loss
            report_metrics = {'vel_loss': vel_loss}
            if self.model.args.use_mask:
                pred_mask = outputs[1]
                mask_loss = self.mask_loss(pred_mask, target_mask)
                mask_acc = accuracy(pred_mask, target_mask)
                loss += self.args.mask_loss_weight * mask_loss
                report_metrics['mask_loss'] = mask_loss
                report_metrics['mask_acc'] = mask_acc
            ade = ADE(pred_pose, target_pose, self.model.args.keypoint_dim)
            fde = FDE(pred_pose, target_pose, self.model.args.keypoint_dim)
            report_metrics['ADE'] = ade
            report_metrics['FDE'] = fde
            self.train_reporter.update(report_metrics, batch_size)

            # backpropagate and optimize
            loss.backward()
            self.optimizer.step()

        self.train_reporter.epoch_finished()
        self.train_reporter.print_values(self.model.args.use_mask, end='| ')

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

        self.valid_reporter.epoch_finished()
        self.valid_reporter.print_values(self.model.args.use_mask)
