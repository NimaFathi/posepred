import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn

from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_model, save_checkpoint
from train.reporter import Reporter


#
# for idx, persons in enumerate(dataloader):
#     print(idx)
#     for (obs_pose, obs_vel, future_pose, future_vel) in persons:
#         print(obs_pose.shape)


class TrainHandler:

    def __init__(self, dataloader_args, model_args, training_args):
        self.dataloader = get_dataloader(dataloader_args)
        self.model = load_model(model_args.load_path) if model_args.load_path else get_model(model_args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=training_args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=training_args.decay_factor,
                                                              patience=training_args.decay_patience, threshold=1e-8,
                                                              verbose=True)
        self.is_multi_person = model_args.is_multi_person
        self.training_args = training_args
        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.device = 'cuda'
        self.dim = self.dataloader.dataset.dim
        self.joints = self.dataloader.dataset.joints

    def train(self):
        train_time_0 = time.time()
        train_s_scores = []
        val_s_scores = []
        for epoch in range(self.training_args.start_epoch, self.training_args.epochs):
            train_reporter = Reporter()

            for persons in self.dataloader:
                if self.is_multi_person:
                    pass
                else:
                    for person_data in persons:
                        for i, data in person_data:
                            person_data[i] = data.to(self.device)

                # (obs_pose, obs_vel, future_pose, future_vel, obs_mask, future_mask)

                batch_size = obs_s.shape[0]
                self.model.zero_grad()

                speed_preds, mask_preds = self.model(pose=obs_pose, vel=obs_s, mask=obs_mask)
                speed_loss = self.l1(speed_preds, target_s)
                mask_loss = self.bce(mask_preds, target_mask)
                mask_acc = mask_accuracy(mask_preds, target_mask)

                preds_p = speed2pos(speed_preds, obs_pose)
                ade_train.update(val=float(ADE_c(preds_p, target_pose)), n=batch_size)
                fde_train.update(val=FDE_c(preds_p, target_pose), n=batch_size)

                loss = 0.8 * speed_loss + 0.2 * mask_loss
                loss.backward()

                optimizer.step()
                avg_epoch_train_speed_loss.update(val=float(speed_loss), n=batch_size)
                avg_epoch_train_mask_loss.update(val=float(mask_loss), n=batch_size)
                avg_epoch_train_mask_acc.update(val=float(mask_acc), n=batch_size)

            if (epoch + 1) % opt.save_freq == 0:
                save_file = os.path.join(
                    opt.save_folder, '{name}_epoch{epoch}.pth'.format(name=opt.name, epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)
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
                    speed_preds, mask_preds = model(pose=obs_pose, vel=obs_s, mask=obs_mask)
                    speed_loss = l1e(speed_preds, target_s)
                    mask_loss = bce(mask_preds, target_mask)
                    mask_acc = mask_accuracy(mask_preds, target_mask)
                    avg_epoch_val_speed_loss.update(val=float(speed_loss), n=batch_size)
                    avg_epoch_val_mask_loss.update(val=float(mask_loss), n=batch_size)
                    avg_epoch_val_mask_acc.update(val=float(mask_acc), n=batch_size)
                    preds_p = speed2pos(speed_preds, obs_pose)
                    ade_val.update(val=float(ADE_c(preds_p, target_pose)), n=batch_size)
                    fde_val.update(val=float(FDE_c(preds_p, target_pose)), n=batch_size)

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
