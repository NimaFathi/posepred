import torch
import torch.nn as nn


class HisRepItselfLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dim = args.keypoint_dim

    def forward(self, model_outputs, input_data):
        future_pose = input_data['future_pose']
        feature_n = future_pose.shape[-1]
        out = model_outputs['pred_pose']
        mpjpe = torch.mean(torch.norm(
            future_pose.reshape(-1, feature_n // self.dim, self.dim) - out.reshape(-1, feature_n // self.dim, self.dim),
            dim=3))

        observed_pose = input_data['observed_pose']
        sup_seq = torch.cat((observed_pose[:, -self.args.kernel_size:, :], future_pose), 1).reshape(-1,
                                                                                                    feature_n // self.dim,
                                                                                                    self.dim)
        out_all = model_outputs['out_all'].reshape(-1, feature_n // self.dim, self.dim)
        loss_all = torch.mean(torch.norm(out_all - sup_seq, dim=3))

        outputs = {'loss': mpjpe, 'loss_all': loss_all}

        if 'pred_mask' in model_outputs.keys():
            mask_loss = self.bce(model_outputs['pred_mask'], input_data['future_mask'])
            outputs['mask_loss'] = mask_loss

        return outputs
