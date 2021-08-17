import os
import sys
import time
import json
from consts import ROOT_DIR

import torch
import torch.optim as optim
import torch.nn as nn

from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, create_new_dir, save_snapshot, load_snapshot
from utils.metrics import accuracy, ADE, FDE
from train.reporter import Reporter


class TrainHandler:

    def __init__(self, args, train_dataloader_args, valid_dataloader_args, model_args=None, load_snapshot_path=None):

        self.args = args
        self.train_dataloader_args = train_dataloader_args
        self.valid_dataloader_args = valid_dataloader_args
        if load_snapshot_path:
            self.model, self.model_args, self.optimizer, epoch, reporters = load_snapshot(load_snapshot_path, args.lr)
            self.args.start_epoch = epoch
            self.snapshots_path = load_snapshot_path[:load_snapshot_path.rindex('/') + 1]
            self.train_reporter, self.valid_reporter = reporters
        else:
            self.model = get_model(model_args)
            self.model_args = model_args
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
            self.snapshots_path = self.create_snapshots_dir()
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

    def train(self):
        self.model.train()
        self.train_reporter.start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            for data in self.train_dataloader:
                if self.train_dataloader.is_multi_person:
                    pass
                else:
                    for i, d in enumerate(data):
                        data[i] = d.to(self.device)
                    batch_size = data[0].shape[0]

                    obs_pose, obs_vel, obs_mask, target_pose, target_vel, target_mask = data
                    self.model.zero_grad()
                    pred_vel, pred_mask = self.model(pose=obs_pose, vel=obs_vel, mask=obs_mask)

                    vel_loss = self.distance_loss(pred_vel, target_vel)
                    mask_loss = self.mask_loss(pred_mask, target_mask)
                    loss = vel_loss + self.args.mask_loss_weight * mask_loss
                    loss.backward()
                    self.optimizer.step()

                    mask_acc = accuracy(pred_mask, target_mask)
                    pred_pose = speed2pos(pred_vel, obs_pose)
                    ade = ADE(pred_pose, target_pose, self.dim)
                    fde = FDE(pred_pose, target_pose, self.dim)
                    self.train_reporter.update([vel_loss, mask_loss, mask_acc, ade, fde], batch_size)

            self.train_reporter.next_epoch()

            if (epoch + 1) % self.args.snapshot_interval == 0:
                save_snapshot(self.model, self.optimizer, self.model_args, epoch + 1, self.train_reporter,
                              self.valid_reporter, self.snapshots_path)

            for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(val_loader):
                obs_s = obs_s.to(device='cuda')
                target_s = target_s.to(device='cuda')
                obs_pose = obs_pose.to(device='cuda')
                target_pose = target_pose.to(device='cuda')
                obs_mask = obs_mask.to(device='cuda')
                target_mask = target_mask.to(device='cuda')
                batch_size = obs_s.shape[0]

                with torch.no_grad():
                    pred_vel, pred_mask = model(pose=obs_pose, vel=obs_s, mask=obs_mask)
                    speed_loss = l1e(pred_vel, target_s)
                    mask_loss = bce(pred_mask, target_mask)
                    mask_acc = mask_accuracy(pred_mask, target_mask)
                    avg_epoch_val_speed_loss.update(val=float(speed_loss), n=batch_size)
                    avg_epoch_val_mask_loss.update(val=float(mask_loss), n=batch_size)
                    avg_epoch_val_mask_acc.update(val=float(mask_acc), n=batch_size)
                    pred_pose = speed2pos(pred_vel, obs_pose)
                    ade_val.update(val=float(ADE_c(pred_pose, target_pose)), n=batch_size)
                    fde_val.update(val=float(FDE_c(pred_pose, target_pose)), n=batch_size)

            val_s_scores.append(avg_epoch_val_speed_loss.avg)
            scheduler.step(avg_epoch_train_speed_loss.avg)
            print('e:', epoch, '| train_speed_loss: %.2f' % avg_epoch_train_speed_loss.avg,
                  '| validation_speed_loss: %.2f' % avg_epoch_val_speed_loss.avg,
                  '| train_mask_loss: %.2f' % avg_epoch_train_mask_loss.avg,
                  '| validation_mask_loss: %.2f' % avg_epoch_val_mask_loss.avg,
                  '| train_mask_acc: %.2f' % avg_epoch_train_mask_acc.avg,
                  '| validation_mask_acc: %.2f' % avg_epoch_val_mask_acc.avg,
                  '| ade_train: %.2f' % ade_train.avg,
                  '| ade_val: %.2f' % ade_val.avg, '| fde_train: %.2f' % fde_train.avg, '| fde_val: %.2f' % fde_val.avg,
                  '| epoch_time.avg:%.2f' % (time.time() - start))
            sys.stdout.flush()
        print("*" * 100)
        print('TRAINING Postrack DONE in:{}!'.format(time.time() - training_start))

    def create_snapshots_dir(self):
        dir_path = create_new_dir(os.path.join(ROOT_DIR, 'exps/train/'))
        with open(dir_path + 'training_args' + '.json', 'w') as f:
            json.dump(self.args, f)
            f.close()
        with open(dir_path + 'train_dataloader_args' + '.json', 'w') as f:
            json.dump(self.train_dataloader_args, f)
            f.close()
        with open(dir_path + 'model_args' + '.json', 'w') as f:
            json.dump(self.model_args, f)
            f.close()
        save_snapshot(self.model, self.optimizer, self.model_args, 0, self.snapshots_path)
        return os.path.join(dir_path + 'snapshots/')
