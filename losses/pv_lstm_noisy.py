import torch
import torch.nn as nn
from metrics import ADE, FDE


class PVLSTMNoisy(nn.Module):

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

        # prediction loss
        pred_vel_loss = self.mse1(model_outputs['pred_vel'], future_vel)

        # completion loss
        comp_vel_loss = self.mse2(model_outputs['comp_vel'], observed_vel)
        comp_ade = ADE(model_outputs['comp_pose'], input_data['observed_pose'], self.args.keypoint_dim)

        loss = (self.args.pred_weight * pred_vel_loss) + (self.args.comp_weight * comp_vel_loss)
        outputs = {'loss': loss, 'pred_vel_loss': pred_vel_loss, 'comp_vel_loss': comp_vel_loss, 'comp_ade': comp_ade,}

        return outputs
