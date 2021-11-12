import torch.nn as nn
from metrics import ADE


class Keyplast(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.mse = nn.MSELoss()

    def forward(self, model_outputs, input_data):
        # completion loss
        comp_ade = ADE(model_outputs['comp_pose'], input_data['observed_pose'], self.args.keypoint_dim)
        comp_ade_noise_only = ADE(model_outputs['comp_pose_noise_only'], input_data['observed_pose'],
                                  self.args.keypoint_dim)

        loss = self.mse(model_outputs['pred_pose'], input_data['future_pose'])
        outputs = {'loss': loss, 'comp_ade': comp_ade,
                   'comp_ade_noise_only': comp_ade_noise_only}

        return outputs
