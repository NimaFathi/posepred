import torch.nn as nn
from metrics import ADE


class PVLSTMComp(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.mse = nn.MSELoss()

    def forward(self, model_outputs, input_data):
        observed_pose = input_data['observed_pose']
        observed_vel = observed_pose[..., 1:, :] - observed_pose[..., :-1, :]

        # completion loss
        comp_vel_loss = self.mse(model_outputs['comp_vel'], observed_vel)
        comp_ade = ADE(model_outputs['comp_pose'], input_data['observed_pose'], self.args.keypoint_dim)

        loss = comp_vel_loss

        outputs = {'loss': loss, 'comp_vel_loss': comp_vel_loss, 'comp_ade': comp_ade}

        return outputs
