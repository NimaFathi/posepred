import time
import torch
import torch.nn as nn
from utils.metrics import accuracy, ADE, FDE
from utils.others import pose_from_vel


class Validator:
    def __init__(self, model, dataloader, reporter, is_interactive, distance_loss):
        self.model = model
        self.dataloader = dataloader
        self.reporter = reporter
        self.is_interactive = is_interactive
        self.distance_loss = nn.L1Loss() if distance_loss == 'L1' else nn.MSELoss()
        self.mask_loss = nn.BCELoss()
        self.device = torch.device('cuda')

    def validate(self):
        self.model.eval()
        time0 = time.time()
        self.validate_()
        print("-" * 100)
        print('Validation is completed in: %.2f' % (time.time() - time0))

    def validate_(self):
        self.reporter.start_time = time.time()
        for data in self.dataloader:
            if self.is_interactive:
                pass
            else:
                for i, d in enumerate(data):
                    data[i] = d.to(self.device)
                batch_size = data[0].shape[0]
                if self.model.args.use_mask:
                    obs_pose, obs_vel, obs_mask, target_pose, target_vel, target_mask = data
                else:
                    obs_pose, obs_vel, target_pose, target_vel = data

                with torch.no_grad():
                    # predict
                    outputs = self.model(data[:len(data) / 2])

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
        self.reporter.print_values()
