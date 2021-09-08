import torch

from utils.losses import MSE


class NearestNeighbor(torch.nn.Module):
    def __init__(self, args):
        super(NearestNeighbor, self).__init__()
        self.args = args
        self.input_size = self.output_size = int(args.keypoints_num * args.keypoint_dim)
        self.distance = MSE()
        self.train_dataloader = None

    def forward(self, inputs):

        in_pose = inputs[0]
        min_distance = None
        best_index = None

        for i, data in enumerate(self.train_dataloader):
            obs_pose = data[0].to('cuda')
            dis = self.distance(in_pose, obs_pose)
            if min_distance is None or dis < min_distance:
                min_distance = dis
                best_index = i

        if self.args.use_mask:
            obs_pose, obs_vel, obs_mask, target_pose, target_vel, target_mask = self.train_dataloader[best_index]
            outputs = [target_vel.to('cuda'), target_mask.to('cuda')]
        else:
            obs_pose, obs_vel, target_pose, target_vel = self.train_dataloader[best_index]
            outputs = [target_vel.to('cuda')]

        return tuple(outputs)
