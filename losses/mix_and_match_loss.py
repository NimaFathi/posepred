import torch.nn as nn
import numpy as np
from losses import KLDivergenceLoss
import numpy as np
import torch.nn as nn

from losses import KLDivergenceLoss


class MixAndMatchLoss(nn.Module):

    def __init__(self, args):
        assert args.x0 is not None, 'x0 should have a float value'
        assert args.k is not None, 'k should have a float value'

        super().__init__()

        self.args = args
        self.kl_loss = KLDivergenceLoss(self.args)
        self.L_skl = nn.MSELoss()
        self.L_rot = nn.MSELoss()
        self.step = 0

    def forward(self, model_outputs, input_data=None):
        pose_loss = self.L_skl(model_outputs['pred_pose'], input_data['future_pose'])
        kl_loss = self.kl_loss(model_outputs)
        kl_weight = self.kl_anneal_function(self.args.anneal_function)
        loss = pose_loss + kl_weight * kl_loss
        outputs = {'loss': loss, 'L_skl': pose_loss, 'L_prior': kl_weight * kl_loss, 'kl_weight': kl_weight}
        return outputs

    def kl_anneal_function(self, anneal_function):
        assert anneal_function in ['logistic', 'linear'], 'anneal_function should be "logistic" or "linear"'
        self.step += 1
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-self.args.k * (self.step - self.args.x0))))
        elif anneal_function == 'linear':
            return min(1, self.step / self.args.x0)
