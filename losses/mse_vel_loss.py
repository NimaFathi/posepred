import torch
import torch.nn as nn


class MSEVelLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, model_outputs, input_data):
        observed_pose = input_data['observed_pose']
        future_pose = input_data['future_pose']
        future_vel = torch.cat(((future_pose[..., 0, :] - observed_pose[..., -1, :]).unsqueeze(-2),
                                future_pose[..., 1:, :] - future_pose[..., :-1, :]), -2)
        vel_loss = self.mse(model_outputs['pred_vel'], future_vel)

        if 'pred_mask' in model_outputs.keys():
            mask_loss = self.bce(model_outputs['pred_mask'], input_data['future_mask'])
        else:
            mask_loss = 0

        loss = vel_loss + (0.3 * mask_loss)

        return loss
