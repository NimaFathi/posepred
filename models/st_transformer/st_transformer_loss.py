
import numpy as np
import torch
import torch.nn as nn
from models.st_transformer.data_proc import Preprocess


class STTransformerLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        # self.pre = Preprocess(args).to('cuda')
        self.args = args
        # self.output_normalizer = nn.BatchNorm1d(66, affine=False, track_running_stats=False)

    def forward(self, y_pred, y_true):
        # y_pred = y_pred['pred_pose']
        # y_pred = y_pred.contiguous().view(-1, 3)
        #
        # y_true = y_true['future_pose']
        # y_true = self.pre(y_true)
        # y_true_2 = y_true.permute(0, 2, 1)
        # y_true_2 = self.output_normalizer(y_true_2)
        # y_true_2 = y_true_2.permute(0, 2, 1)
        # y_true_2 = y_true_2.contiguous().view(-1, 3)
        # y_true = y_true.contiguous().view(-1, 3)
        # # print(torch.mean(torch.norm(y_pred - y_true, dim=-1)))
        # return {'loss': torch.mean(torch.norm(y_pred - y_true_2, dim=-1))}

        y_pred = y_pred['pred_pose']
        y_pred = y_pred.contiguous().view(-1, 3)

        y_true_pred = y_true['future_pose']
        # y_true_obs = self.pre(y_true['observed_pose'])

        # y_true_pred -= torch.mean(y_true_obs, dim=[0, 1]).unsqueeze(0).unsqueeze(0)
        # y_true_pred /= torch.std(y_true_obs, dim=[0, 1]).unsqueeze(0).unsqueeze(0)

        y_true_pred = y_true_pred.contiguous().view(-1, 3)

        return {'loss': torch.mean(torch.norm(y_pred - y_true_pred, dim=-1))}