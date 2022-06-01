
import numpy as np
import torch
import torch.nn as nn
from models.st_transformer.data_proc import Preprocess


class STTransformerLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, y_pred, y_true):

        batch = y_true

        y_pred = y_pred['pred_pose']
        y_true = y_true['future_pose']

        B,T,JC = y_pred.shape
        C = 3
        J = JC//C

        y_pred = y_pred.view(B, T, J, C)
        y_true = y_true.view(B, T, J, C)

        l = torch.norm(y_pred - y_true, dim=-1) # B,T,J

        if "reconstruction_mask" in batch.keys():
            l *= torch.mean(batch['reconstruction_mask'].view(B, T, J, C), dim=-1)

        return {'loss': torch.mean(l)}