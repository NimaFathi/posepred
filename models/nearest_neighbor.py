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
        best_pred_pose = None
        best_pred_mask = None

        obs_pose = inputs['observed_pose']
        assert obs_pose.shape[0] == 1, "only support batch_size 1 in nearest neighbor"
        in_vel = (obs_pose[..., 1:, :] - obs_pose[..., :-1, :]).view(1, -1)

        for data in self.train_dataloader:
            pose = inputs['observed_pose']
            obs_vel = pose[..., 1:, :] - pose[..., :-1, :]
            bs = obs_vel.shape[0]
            dis = self.distance(in_vel.repeat(bs, 1), obs_vel.view(bs, -1)).sum(1)
            value, ind = torch.min(dis, 0, out=None)
            if min_distance is None or value < min_distance:
                min_distance = value
                if self.args.use_mask:
                    best_pred_pose = data['future_pose'][ind]
                    best_pred_mask = data['future_mask'][ind]
                else:
                    best_pred_pose = data['future_pose'][ind]

        outputs = {'pred_pose': best_pred_pose.unsqueeze(0)}

        if self.args.use_mask:
            outputs['pred_mask'] = best_pred_mask.unsqueeze(0)

        return outputs
