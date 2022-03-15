import numpy as np
import torch
import torch.nn as nn


class STSGCNLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                                  26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                  46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                                  75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        # joints at same loc
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

        self.index_to_ignore = index_to_ignore
        self.index_to_equal = index_to_equal

    def forward(self, y_pred, y_true):
        y_pred_real = y_pred['pred_xyz_pose']
        y_pred_real = y_pred_real.view((-1, y_pred_real.shape[-1]))

        y_true = y_true['future_xyz_pose']
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], y_true.shape[2], y_true.shape[3]).view(
            y_true.shape[0] * y_true.shape[1], -1)

        y_pred = y_true.clone()
        y_pred[:, self.dim_used] = y_pred_real
        y_pred[:, self.index_to_ignore] = y_pred[:, self.index_to_equal]

        loss = torch.mean(torch.norm(y_true.contiguous().view(-1, 3) - y_pred.contiguous().view(-1, 3), 2, 1))
        outputs = {'loss': loss}

        return outputs
