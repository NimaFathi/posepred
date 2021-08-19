import time
import torch
import torch.nn as nn
from utils.save_load import save_snapshot
from utils.metrics import accuracy, ADE, FDE
from utils.others import pose_from_vel


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
        self.distance_loss = nn.L1Loss() if self.args.distance_loss == 'L1' else nn.MSELoss()
        self.mask_loss = nn.BCELoss()
        self.device = torch.device('cuda')

    def train(self):
        self.model.train()
        time0 = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.train_()
            self.validate_()
            self.scheduler.step(self.valid_reporter.history['vel_loss'][-1])
            if (epoch + 1) % self.args.snapshot_interval == 0:
                save_snapshot(self.model, self.optimizer, self.args.lr, epoch + 1, self.train_reporter,
                              self.valid_reporter, self.args.save_dir)
        print("-" * 100)
        print('Training is completed in: %.2f' % (time.time() - time0))

    def train_(self):
        self.train_reporter.start_time = time.time()
        for data in self.train_dataloader:
            if self.args.is_interactive:
                pass
            else:
                for i, d in enumerate(data):
                    data[i] = d.to(self.device)
                batch_size = data[0].shape[0]

                if self.model_args.use_mask:
                    obs_pose, obs_vel, obs_mask, target_pose, target_vel, target_mask = data
                    model_input = (obs_pose, obs_vel, obs_mask)
                else:
                    obs_pose, obs_vel, target_pose, target_vel = data
                    model_input = (obs_pose, obs_vel)

                # predict
                self.model.zero_grad()
                pred_vel, pred_mask = self.model(model_input)

                # calculate metrics
                vel_loss = self.distance_loss(pred_vel, target_vel)
                mask_loss = self.mask_loss(pred_mask, target_mask)
                mask_acc = accuracy(pred_mask, target_mask)
                pred_pose = pose_from_vel(pred_vel, obs_pose[..., -1, :])
                ade = ADE(pred_pose, target_pose, self.model_args.keypoint_dim)
                fde = FDE(pred_pose, target_pose, self.model_args.keypoint_dim)
                self.train_reporter.update([vel_loss, mask_loss, mask_acc, ade, fde], batch_size)

                # calculate loss and backpropagate
                loss = vel_loss + self.args.mask_loss_weight * mask_loss
                loss.backward()
                self.optimizer.step()

        self.train_reporter.epoch_finished()
        self.train_reporter.print_values()

    def validate_(self):
        self.train_reporter.start_time = time.time()
        for data in self.valid_dataloader:
            if self.args.is_interactive:
                pass
            else:
                for i, d in enumerate(data):
                    data[i] = d.to(self.device)
                batch_size = data[0].shape[0]
                obs_pose, obs_vel, obs_mask, target_pose, target_vel, target_mask = data

                with torch.no_grad():
                    # predict
                    pred_vel, pred_mask = self.model(pose=obs_pose, vel=obs_vel, mask=obs_mask)

                    # calculate metrics
                    vel_loss = self.distance_loss(pred_vel, target_vel)
                    mask_loss = self.mask_loss(pred_mask, target_mask)
                    mask_acc = accuracy(pred_mask, target_mask)
                    pred_pose = pose_from_vel(pred_vel, obs_pose)
                    ade = ADE(pred_pose, target_pose, self.model_args.keypoint_dim)
                    fde = FDE(pred_pose, target_pose, self.model_args.keypoint_dim)
                    self.valid_reporter.update([vel_loss, mask_loss, mask_acc, ade, fde], batch_size)

        self.valid_reporter.epoch_finished()
        self.valid_reporter.print_values()
