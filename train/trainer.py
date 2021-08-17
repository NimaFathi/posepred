import os
import time
import json
from consts import ROOT_DIR

import torch
import torch.optim as optim
import torch.nn as nn

from data_loader.data_loader import get_dataloader
from train.reporter import Reporter
from utils.save_load import get_model, create_new_dir, save_snapshot, load_snapshot
from utils.metrics import accuracy, ADE, FDE
from utils.others import pose_from_vel


class Trainer:

    def __init__(self, args, train_dataloader_args, valid_dataloader_args, model_args=None, load_snapshot_path=None):
        self.args = args
        if load_snapshot_path:
            self.model, self.model_args, self.optimizer, epoch, reporters = load_snapshot(load_snapshot_path, args.lr)
            self.args.start_epoch = epoch
            self.snapshots_dir_path = load_snapshot_path[:load_snapshot_path.rindex('/') + 1]
            self.train_reporter, self.valid_reporter = reporters
        else:
            self.model = get_model(model_args)
            self.model_args = model_args
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
            self.snapshots_dir_path = self.create_snapshots_dir(train_dataloader_args, valid_dataloader_args)
            self.train_reporter = Reporter()
            self.valid_reporter = Reporter()

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=args.decay_factor,
                                                              patience=args.decay_patience, threshold=1e-8,
                                                              verbose=True)
        self.device = torch.device('cuda')
        self.distance_loss = nn.L1Loss() if args.distance_loss == 'L1' else nn.MSELoss()
        self.mask_loss = nn.BCELoss()

        self.train_dataloader = get_dataloader(train_dataloader_args)
        self.valid_dataloader = get_dataloader(valid_dataloader_args)
        self.dim = train_dataloader_args.data_dim
        self.use_mask = train_dataloader_args.use_mask
        self.is_multi_person = False

    def train(self):

        self.model.train()
        time0 = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.train_()
            self.validate_()
            self.scheduler.step(self.valid_reporter.history['vel_loss'][-1])

            if (epoch + 1) % self.args.snapshot_interval == 0:
                save_snapshot(self.model, self.optimizer, self.model_args, epoch + 1, self.train_reporter,
                              self.valid_reporter, self.snapshots_dir_path)

        print("-" * 100)
        print('Training is completed in: %.2f' % (time.time() - time0))

    def train_(self):
        self.train_reporter.start_time = time.time()
        for data in self.train_dataloader:
            if self.is_multi_person:
                pass
            else:
                for i, d in enumerate(data):
                    data[i] = d.to(self.device)
                batch_size = data[0].shape[0]
                obs_pose, obs_vel, obs_mask, target_pose, target_vel, target_mask = data

                # predict
                self.model.zero_grad()
                pred_vel, pred_mask = self.model(pose=obs_pose, vel=obs_vel, mask=obs_mask)

                # calculate metrics
                vel_loss = self.distance_loss(pred_vel, target_vel)
                mask_loss = self.mask_loss(pred_mask, target_mask)
                mask_acc = accuracy(pred_mask, target_mask)
                pred_pose = pose_from_vel(pred_vel, obs_pose[..., -1, :])
                ade = ADE(pred_pose, target_pose, self.dim)
                fde = FDE(pred_pose, target_pose, self.dim)
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
            if self.is_multi_person:
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
                    ade = ADE(pred_pose, target_pose, self.dim)
                    fde = FDE(pred_pose, target_pose, self.dim)
                    self.valid_reporter.update([vel_loss, mask_loss, mask_acc, ade, fde], batch_size)

        self.valid_reporter.epoch_finished()
        self.valid_reporter.print_values()

    def create_snapshots_dir(self, train_dataloader_args, valid_dataloader_args):
        dir_path = create_new_dir(os.path.join(ROOT_DIR, 'exps/train/'))
        with open(dir_path + 'training_args' + '.json', 'w') as f:
            json.dump(self.args, f)
            f.close()
        with open(dir_path + 'model_args' + '.json', 'w') as f:
            json.dump(self.model_args, f)
            f.close()
        with open(dir_path + 'train_dataloader_args' + '.json', 'w') as f:
            json.dump(train_dataloader_args, f)
            f.close()
        with open(dir_path + 'valid_dataloader_args' + '.json', 'w') as f:
            json.dump(valid_dataloader_args, f)
            f.close()
        save_snapshot(self.model, self.optimizer, self.model_args, 0, self.train_reporter, self.valid_reporter,
                      self.snapshots_dir_path)
        return os.path.join(dir_path + 'snapshots/')
