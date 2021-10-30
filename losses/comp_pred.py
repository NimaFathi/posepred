import torch
import torch.nn as nn
from metrics import ADE


class CompPred(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.mse1 = nn.MSELoss()
        self.mse2 = nn.MSELoss()

    def forward(self, model_outputs, input_data):
        # prediction loss
        pred_pose_loss = self.mse1(model_outputs['pred_pose'], input_data['future_pose'])

        # completion loss
        comp_pose_loss = self.mse2(model_outputs['comp_pose'], input_data['observed_pose'])
        comp_ade = ADE(model_outputs['comp_pose'], input_data['observed_pose'], self.args.keypoint_dim)

        # KL_Divergence loss
        kl_loss = -0.5 * torch.sum(1 + model_outputs['std'] - model_outputs['mean'].pow(2) - model_outputs['std'].exp())

        loss = (self.args.pred_weight * pred_pose_loss) + (self.args.comp_weight * comp_pose_loss) + (
                self.args.kl_weight * kl_loss)
        outputs = {'loss': loss, 'pred_pose_loss': pred_pose_loss, 'comp_pose_loss': comp_pose_loss, 'kl_loss': kl_loss,
                   'comp_ade': comp_ade}

        return outputs
