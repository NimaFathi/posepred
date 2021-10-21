import torch
import torch.nn as nn
from metrics import ADE


class CompPredCenter(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.mse1 = nn.MSELoss()
        self.mse2 = nn.MSELoss()

    def forward(self, model_outputs, input_data):
        observed_pose = input_data['observed_pose']
        future_pose = input_data['future_pose']

        # prediction loss
        future_pose_center = future_pose - observed_pose[:, 0:1, :].repeat(1, future_pose.shape[1], 1)
        pred_pose_loss = self.mse1(model_outputs['pred_pose_center'], future_pose_center)

        # completion loss
        bs, frames_n, featurs_n = observed_pose.shape
        obs_pose_center = observed_pose - observed_pose[:, 0:1, :].repeat(1, frames_n, 1)
        noise = model_outputs['noise'].reshape(bs, frames_n, -1)
        final_comp = torch.where(noise == 1, model_outputs['comp_pose_center'], obs_pose_center)
        comp_pose_loss = self.mse2(final_comp, obs_pose_center)
        comp_pose_ade = ADE(model_outputs['comp_pose'], input_data['observed_pose'], self.args.keypoint_dim)

        # KL_Divergence loss
        kl_loss = -0.5 * torch.sum(1 + model_outputs['std'] - model_outputs['mean'].pow(2) - model_outputs['std'].exp())

        loss = (self.args.pred_weight * pred_pose_loss) + (self.args.comp_weight * comp_pose_loss) + (
                self.args.kl_weight * kl_loss)
        outputs = {'loss': loss, 'pred_pose_loss': pred_pose_loss, 'comp_pose_loss': comp_pose_loss, 'kl_loss': kl_loss,
                   'comp_pose_ade': comp_pose_ade}

        return outputs
