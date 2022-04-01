import numpy as np
import torch
import torch.nn as nn


class SphericalLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

    def forward(self, y_pred, y_true):
        y_pred = y_pred['pred_pose']
        y_true = y_true['future_pose']
        assert y_pred.shape == y_true.shape
        B, T, D = y_pred.shape
        y_pred = y_pred.reshape(B, T, D//3, 3)[:, :, :, 1:].reshape(B*T, -1)
        y_true = y_true.reshape(B, T, D//3, 3)[:, :, :, 1:].reshape(B*T, -1)
        
        loss = torch.mean(torch.norm(y_true.contiguous().view(-1, 2) - y_pred.contiguous().view(-1, 2), 2, 1))
        outputs = {'loss': loss}

        return outputs