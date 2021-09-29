import torch
from torch import nn

from utils.others import pose_from_vel


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

        obs_pose = inputs['observed_pose']
        in_vel = (obs_pose[..., 1:, :] - obs_pose[..., :-1, :]).view(1, -1)
        assert in_vel.shape[0] == 1, "only support batch_size 1 in nearest neighbor"

        for data in self.train_dataloader:
            obs_vel = data[1].to('cuda')
            bs = obs_vel.shape[0]
            dis = self.distance(in_vel.repeat(bs, 1), obs_vel.view(bs, -1)).sum(1)
            value, ind = torch.min(dis, 0, out=None)
            if min_distance is None or value < min_distance:
                min_distance = value
                if self.args.use_mask:
                    best_pred_vel = data[4][ind]
                    best_pred_mask = data[5][ind]
                else:
                    best_pred_vel = data[3][ind]

        pred_vel = best_pred_vel.unsqueeze(0).to('cuda')
        pred_pose = pose_from_vel(pred_vel, obs_pose[..., -1, :])
        outputs = {'pred_pose': pred_pose, 'pred_vel': pred_vel}

        if self.args.use_mask:
            pred_mask = best_pred_mask.unsqueeze(0).to('cuda')
            outputs['pred_mask'] = pred_mask

        return outputs
