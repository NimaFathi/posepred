import numpy as np
import torch.nn as nn

from losses import KLDivergenceLoss
from utils.others import qeuler


class MixAndMatchLoss(nn.Module):

    def __init__(self, args):
        assert args.x0 is not None, 'x0 should have a float value'
        assert args.k is not None, 'k should have a float value'

        super().__init__()

        self.args = args
        self.kl_loss = KLDivergenceLoss(self.args)
        self.elbo = ELBOMixAndMatch(self.args)
        self.L_rot = nn.MSELoss()
        self.L_skl = nn.MSELoss()
        self.L_rot = nn.MSELoss()
        self.step = 0

    def forward(self, model_outputs, input_data=None):
        q_target_pose = qeuler(q=input_data['future_pose'], order='xyz')
        rot_loss = self.L_rot(model_outputs['pred_q_pose'], q_target_pose)
        pose_loss = self.L_skl(model_outputs['pred_pose'], input_data['future_pose'])
        kl_loss = self.kl_loss(model_outputs)
        kl_weight = self.kl_anneal_function(self.args.anneal_function)
        elbo_loss = self.elbo(model_outputs, input_data)
        loss = rot_loss + pose_loss + kl_weight * kl_loss
        outputs = {'loss': loss, 'L_rot': rot_loss, 'L_skl': pose_loss, 'L_prior': kl_weight * kl_loss,
                   'kl_weight': kl_weight, 'ELBO': elbo_loss}
        return outputs

    def kl_anneal_function(self, anneal_function):
        assert anneal_function in ['logistic', 'linear'], 'anneal_function should be "logistic" or "linear"'
        self.step += 1
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-self.args.k * (self.step - self.args.x0))))
        elif anneal_function == 'linear':
            return min(1, self.step / self.args.x0)


class ELBOMixAndMatch(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.bce = nn.BCELoss(reduction='sum')
        self.kl_loss = KLDivergenceLoss(self.args)

    def forward(self, model_outputs, input_data=None):
        kl_loss = self.kl_loss(model_outputs)
        bce_loss = self.bce(model_outputs['pred_pose'], input_data['target_pose'])

        return kl_loss + bce_loss
