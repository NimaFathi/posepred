import torch
import torch.nn as nn


class CompPredVel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.mse1 = nn.MSELoss()
        self.mse2 = nn.MSELoss()

    def forward(self, model_outputs, input_data):
        observed_pose = input_data['observed_pose']
        future_pose = input_data['future_pose']
        observed_vel = observed_pose[..., 1:, :] - observed_pose[..., :-1, :]
        future_vel = torch.cat(((future_pose[..., 0, :] - observed_pose[..., -1, :]).unsqueeze(-2),
                                future_pose[..., 1:, :] - future_pose[..., :-1, :]), -2)

        pred_loss = self.mse1(model_outputs['pred_vel'], future_vel)

        bs, frames_n, featurs_n = observed_vel.shape
        mask = model_outputs['mask'].reshape(bs, frames_n, -1)
        completed = torch.where(mask == 1, model_outputs['completed_vel'], observed_vel)
        completion_loss = self.mse2(completed, observed_vel)

        loss = self.args.pred_weight * pred_loss + self.args.comp_weight * completion_loss
        outputs = {'loss': loss, 'pred_loss': pred_loss, 'completion_loss': completion_loss}

        return outputs
