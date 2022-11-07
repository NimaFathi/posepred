
import numpy as np
import torch
import torch.nn as nn
from losses.hua_loss import HUALoss
from models.st_transformer.data_proc import Preprocess, Human36m_Preprocess, AMASS_3DPW_Preprocess


def smooth(src):
    """
    data:[bs, 60, 96]
    """
    smooth_data = src.clone()
    for i in range(src.shape[1]):
        smooth_data[:, i] = torch.mean(src[:, :i+1], dim=1)
    return smooth_data


class CSDI_PGBIG_HUALoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        print(args.stages)

        if args.inner_type == "HUAL":
            if 'S' in args.tasks:
                self.hual = [
                    HUALoss(args).to(args.device)
                    for _ in range(args.stages)
                ]
            else:
                self.hual = HUALoss(args).to(args.device)


    def forward(self, y_pred, y_true):

        y_future = y_true['future_pose']

        B, T, JC = y_future.shape
        J, C = JC // 3, 3

        smooth_s = [y_future,]
        for _ in range(self.args.stages - 1):
            smooth_s.append(smooth(smooth_s[-1]))

        losses = []
        if self.args.inner_type == "HUAL":
            if 'S' in self.args.tasks:
                for i in range(self.args.stages):
                    losses.append(
                        self.hual[i]({'pred_pose': y_pred['pred_stage'][i]}, 
                                     {'future_pose': smooth_s[-i]})['loss'])
            else:
                for i in range(self.args.stages):
                    losses.append(
                        self.hual({'pred_pose': y_pred['pred_stage'][i]}, 
                                     {'future_pose': smooth_s[-i]})['loss'])

        else:
            for i in range(self.args.stages):
                losses.append(
                    torch.mean(torch.norm(
                        y_pred['pred_stage'][i].view(B, T, J, C) - smooth_s[-i].view(B, T, J, C)
                        , dim=-1))
                    )

        return {
            'loss': sum(losses) / self.args.stages
        }

