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
        loss = torch.mean(torch.norm(y_true.contiguous().view(-1, 3) - y_pred.contiguous().view(-1, 3), 2, 1))
        outputs = {'loss': loss}

        return outputs