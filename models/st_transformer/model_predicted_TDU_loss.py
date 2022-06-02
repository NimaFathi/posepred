
import numpy as np
import torch
import torch.nn as nn

dim_used = np.array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
       26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
       46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
       63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
       80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95])



class model_predicted_TDUncertaintyLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args


    def proc(self, observed_pose):
        return observed_pose[:, :, dim_used]


    def forward(self, y_pred, y_true):
      
        sigma = self.proc(y_pred['sigma']) # B,T,JC
        y_pred = self.proc(y_pred['pred_pose']) # B,T,JC
        y_true = self.proc(y_true['future_pose']) # B,T,JC

        B,T,JC = y_pred.shape
        C = 3
        J = JC//C

        sigma = sigma.view(B, T, J, C)
        sigma = torch.mean(sigma, dim=-1) # B, T, J
        sigma = sigma + 3.5

        sigma = torch.clamp(sigma, min=-10.0)
        y_pred = y_pred.view(B, T, J, C)
        y_true = y_true.view(B, T, J, C)

        l = torch.norm(y_pred - y_true, dim=-1) # B,T,J

        l = torch.mean(torch.exp(-sigma) * l + sigma)

        return {
          'loss' : l
        }

