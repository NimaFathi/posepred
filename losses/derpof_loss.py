import torch
import torch.nn as nn


class DeRPoFLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, model_outputs, input_data):
        observed_pose = input_data['observed_pose']
        future_pose = input_data['future_pose']
        future_vel = torch.cat(((future_pose[..., 0, :] - observed_pose[..., -1, :]).unsqueeze(-2),
                                future_pose[..., 1:, :] - future_pose[..., :-1, :]), -2)
        frames_num, bs, _ = future_vel.shape

        # global velocity
        future_vel_global = 0.5 * (
                future_vel.view(frames_num, bs, self.keypoints_num, self.keypoint_dim)[:, :, 0] + future_vel.view(
            frames_num, bs, self.keypoints_num, self.keypoint_dim)[:, :, 1])

        # local velocity
        future_vel_local = (
                future_vel.view(frames_num, bs, self.keypoints_num, self.keypoint_dim) - future_vel_global.view(
            frames_num, bs, 1, self.keypoint_dim)).view(frames_num, bs, self.features_num)

        loss_global = self.mse(future_vel_global, model_outputs['pred_vel_global'])
        loss_local = vae_loss_function(future_vel_local, model_outputs['pred_vel_local'], model_outputs['mean'],
                                       model_outputs['log_var'])

        loss = loss_global + 0.1 * loss_local

        outputs = {'loss': loss, 'global_loss': loss_global, 'local_loss': loss_local}

        if 'pred_mask' in model_outputs.keys():
            mask_loss = self.bce(model_outputs['pred_mask'], input_data['future_mask'])
            outputs['mask_loss'] = mask_loss

        return outputs


def vae_loss_function(x, x_hat, mean, log_var):
    assert x_hat.shape == x.shape
    reconstruction_loss = torch.mean(torch.norm(x - x_hat, dim=len(x.shape) - 1))
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + 0.01 * KLD
