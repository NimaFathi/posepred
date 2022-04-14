from ast import GtE
import torch
from torch import nn
import sys

class AELoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args=args 
    def forward(self, model_outputs, input_data):
        b, t, j = input_data["future_pose"].shape
        gt = torch.cat([input_data["observed_pose"].clone(), input_data["future_pose"].clone()], dim=1)
        gt = gt.reshape(-1, gt.shape[-1])
        out = model_outputs["out"]
        loss = nn.MSELoss()(gt, out)
        
        return {"loss":loss}