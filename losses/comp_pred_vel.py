import torch
import torch.nn as nn
from metrics import ADE


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

        # prediction loss
        pred_vel_loss = self.mse1(model_outputs['pred_vel'], future_vel)

        # completion loss
        bs, frames_n, featurs_n = observed_vel.shape
        noise = model_outputs['noise'].reshape(bs, frames_n, -1)
        final_comp = torch.where(noise == 1, model_outputs['comp_vel'], observed_vel)
        comp_vel_loss = self.mse2(final_comp, observed_vel)
        comp_pose_ade = ADE(model_outputs['comp_pose'], input_data['observed_pose'][:, 1:, :], self.args.keypoint_dim)

        # KL_Divergence loss
        kl_loss = -0.5 * torch.sum(1 + model_outputs['std'] - model_outputs['mean'].pow(2) - model_outputs['std'].exp())

        loss = (self.args.pred_weight * pred_vel_loss) + (self.args.comp_weight * comp_vel_loss) + (
                self.args.kl_weight * kl_loss)
        outputs = {'loss': loss, 'pred_vel_loss': pred_vel_loss, 'comp_vel_loss': comp_vel_loss, 'kl_loss': kl_loss,
                   'comp_pose_ade': comp_pose_ade}

        return outputs
