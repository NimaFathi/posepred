import os
import sys
import time
from consts import ROOT_DIR

import torch
import torch.optim as optim
import torch.nn as nn

from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_model, create_new_dir
from utils.metrics import accuracy, ADE, FDE
from train.reporter import Reporter


class TrainHandler:

    def __init__(self, dataloader_args, model_args, training_args):
        self.dataloader = get_dataloader(dataloader_args)
        self.model = load_model(model_args.load_path) if model_args.load_path else get_model(model_args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=training_args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=training_args.decay_factor,
                                                              patience=training_args.decay_patience, threshold=1e-8,
                                                              verbose=True)

        self.device = 'cuda'
        self.distance_loss = nn.L1Loss() if training_args.distance_loss == 'L1' else nn.MSELoss()
        self.mask_loss = nn.BCELoss()
        self.args = training_args
        self.dim = self.dataloader.dataset.dim
        self.joints = self.dataloader.dataset.joints
        self.is_multi_person = dataloader_args.is_multi_person

    def train(self):

        train_time_0 = time.time()
        train_s_scores = []
        val_s_scores = []

        for epoch in range(self.args.start_epoch, self.args.epochs):

            train_reporter = Reporter()
            for data in self.dataloader:
                if self.is_multi_person:
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
                    train_reporter.update([vel_loss, mask_loss, mask_acc, ade, fde], batch_size)

            if (epoch + 1) % self.args.save_interval == 0:
                dir_path = create_new_dir(os.path.join(ROOT_DIR, 'exps/test/'))
                save_path = os.path.join(dir_path, '%03d.pt' % (epoch + 1))
                save_model(self.model, self.optimizer, opt, epoch, save_path)

            train_s_scores.append(avg_epoch_train_speed_loss.avg)

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
