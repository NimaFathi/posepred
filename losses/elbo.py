import torch
import torch.nn as nn


class ELBO(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.bce = nn.BCELoss(reduction='sum')
        nn.BCELoss()

    def forward(self, model_outputs, input_data=None):
        kl_loss = -0.5 * torch.sum(
            1 + model_outputs['sigma'] - model_outputs['mu'].pow(2) - model_outputs['sigma'].exp())
        bce_loss = self.bce(model_outputs['pred_pose'], input_data['target_pose'])
        outputs = {'loss': kl_loss + bce_loss}
        return outputs
