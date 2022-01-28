import torch
import torch.nn as nn


class MPJPE(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

    def forward(self, y_pred, y_true):
        y_pred = y_pred['pred_pose'].contiguous().view(-1, 3)
        y_true = y_true['future_pose'].contiguous().view(-1, 3)

        loss = torch.mean(torch.norm(y_true - y_pred, 2, 1))
        outputs = {'loss': loss}

        return outputs
