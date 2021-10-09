import torch
import torch.nn as nn


class KLDivergenceLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

    def forward(self, model_outputs, input_data=None):
        kl_loss = -0.5 * torch.sum(
            1 + model_outputs['sigma'] - model_outputs['mu'].pow(2) - model_outputs['sigma'].exp())
        outputs = {'loss': kl_loss}
        return outputs
