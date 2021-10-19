import torch
import torch.nn as nn
from metrics import ADE


class CompPredPose(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.mse1 = nn.MSELoss()
        self.mse2 = nn.MSELoss()

    def forward(self, model_outputs, input_data):
        observed_pose = input_data['observed_pose']
        future_pose = input_data['future_pose']

        pred_pose_loss = self.mse1(model_outputs['pred_pose'], future_pose)

        bs, frames_n, featurs_n = observed_pose.shape
        mask = model_outputs['mask'].reshape(bs, frames_n, -1)
        final_comp = torch.where(mask == 1, model_outputs['comp_pose'], observed_pose)
        comp_pose_loss = self.mse2(final_comp, observed_pose)

        comp_pose_ade = ADE(model_outputs['comp_pose'], input_data['ovserved_pose'][:, 1:, :], self.args.keypoint_dim)

        loss = self.args.pred_weight * pred_pose_loss + self.args.comp_weight * comp_pose_loss
        outputs = {'loss': loss, 'pred_pose_loss': pred_pose_loss, 'comp_pose_loss': comp_pose_loss,
                   'comp_pose_ade': comp_pose_ade}

        return outputs
