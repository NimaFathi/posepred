import numpy as np
import torch
import torch.nn as nn


class SphericalLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

    def forward(self, y_pred, y_true):
        y_pred = y_pred['pred_pose']
        y_pred = y_pred.view((-1, y_pred.shape[-1]))

        y_true = y_true['future_pose']
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1) # BT, JD

        vel_true = y_true[..., 1:, :] - y_true[..., :-1, :]
        vel_pred = y_pred[..., 1:, :] - y_pred[..., :-1, :]

        pose_loss = torch.mean(torch.norm(y_true.contiguous().view(-1, 3) - y_pred.contiguous().view(-1, 3), 2, 1))
        vel_loss = torch.mean(torch.norm(vel_true.contiguous().view(-1, 3) - vel_pred.contiguous().view(-1, 3), 2, 1))
        loss = pose_loss + vel_loss
        outputs = {'loss': loss, 'pose_loss': pose_loss, 'vel_loss': vel_loss}

        return outputs