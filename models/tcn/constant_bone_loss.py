
import numpy as np
import torch
import torch.nn as nn
from models.tcn.data_proc_constant_bone import Preprocess


class CBLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pre = Preprocess().to('cuda')
        self.args = args
        self.nf = 1

    def forward(self, y_pred, y_true):
        y_pred = y_pred['pred_pose']
        y_pred = y_pred.contiguous().view(-1, 3)

        y_true = y_true['future_pose']
        # y_true = self.pre(y_true)
        y_true = y_true.contiguous().view(-1, 3)

        return {'loss': torch.mean(torch.norm(y_pred - y_true, dim=-1))}

        # loss = torch.mean(torch.abs(7 - torch.norm(y_pred, dim=-1)))

        # # print(torch.mean(torch.norm(y_pred, dim=-1)))
        #
        # y_pred = y_pred / torch.unsqueeze(torch.norm(y_pred, dim=-1), -1)
        # y_true = y_true / torch.unsqueeze(torch.norm(y_true, dim=-1), -1)
        #
        # loss = -torch.mean(y_pred * y_true)
        #
        # outputs = {'loss': loss}

        return outputs