import torch
import torch.nn as nn
from losses import KLDivergenceLoss


class ELBO(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.kl_loss = KLDivergenceLoss(self.args)
        self.log_scale = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, model_outputs, input_data=None):
        kl_loss = self.kl_loss(model_outputs)
        recon_loss = self.gaussian_likelihood(model_outputs['pred_pose'], self.log_scale,
                                              input_data['future_pose'])
        elbo = kl_loss['loss'] - recon_loss
        return elbo.mean()

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2))
