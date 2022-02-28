import torch
import torch.nn as nn
import torch.nn.functional as F

class TRANS_CVAE(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

    def forward(self, output, input_data):
        observed_pose = input_data['observed_pose']
        comp_pose = output['comp_pose']

        # completion loss
        comp_pose_loss = F.mse_loss(observed_pose, comp_pose, reduction='mean')

        # KL_Divergence loss
        mu, logvar = output["mu"], output["logvar"]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = (self.args.comp_weight * comp_pose_loss) + (self.args.kl_weight * kl_loss)
        outputs = {'loss': loss, 'comp_pose_loss': comp_pose_loss, 'kl_loss': kl_loss}

        return outputs
