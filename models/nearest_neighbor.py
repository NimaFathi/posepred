import torch
from torch import nn


class NearestNeighbor(nn.Module):
    def __init__(self, args):
        super(NearestNeighbor, self).__init__()
        self.args = args
        self.distance = nn.MSELoss(reduction='none')
        self.train_dataloader = None

    def forward(self, inputs):

        min_distance = None
        best_pred_vel = None
        best_pred_mask = None

        in_vel = inputs[1]
        assert in_vel.shape[0] == 1, "only support batch_size 1 in nearest neighbor"

        for data in self.train_dataloader:
            obs_vel = data[1].to('cuda')
            bs = obs_vel.shape[0]
            in_vel = in_vel.view(1, -1).repeat(bs, 1)
            dis = self.distance(in_vel, obs_vel.view(bs, -1)).sum(1)
            value, ind = torch.min(dis, 0, out=None)
            if min_distance is None or value < min_distance:
                min_distance = value
                if self.args.use_mask:
                    best_pred_vel = data[4][ind]
                    best_pred_mask = data[5][ind]
                else:
                    best_pred_vel = data[3][ind]

        if self.args.use_mask:
            outputs = [best_pred_vel.unsqueeze(0).to('cuda'), best_pred_mask.unsqueeze(0).to('cuda')]
        else:
            outputs = [best_pred_vel.unsqueeze(0).to('cuda')]

        return tuple(outputs)
