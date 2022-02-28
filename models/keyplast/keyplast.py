import torch
import torch.nn as nn


class Keyplast(nn.Module):
    def __init__(self, args):
        super(Keyplast, self).__init__()
        self.args = args

    def forward(self, inputs):
        pose = inputs['observed_pose']

        if 'observed_noise' not in inputs.keys():
            raise Exception("This model requires noise. set is_noisy to True")

        # zero-vel prediction
        last_frame = pose[..., -1, :].unsqueeze(-2)
        ndims = len(pose.shape)
        pred_pose = last_frame.repeat([1 for _ in range(ndims - 2)] + [self.args.pred_frames_num, 1])

        # completion
        bs, frames_n, features_n = pose.shape
        pose_noise = inputs['observed_noise'].reshape(bs, frames_n, self.args.keypoints_num, 1)
        pose_noise = pose_noise.repeat(1, 1, 1, self.args.keypoint_dim).reshape(bs, frames_n, features_n)
        comp_pose = torch.clone(pose)
        for f in range(1, frames_n):
            comp_pose[:, f] = torch.where(pose_noise[:, f] == 1, pose[:, f-1], pose[:, f])

        bs, frames_n, feat = comp_pose.shape
        comp_pose_noise_only = torch.where(pose_noise.view(bs, frames_n, feat) == 1, comp_pose, pose)

        outputs = {'pred_pose': pred_pose, 'comp_pose': comp_pose, 'comp_pose_noise_only': comp_pose_noise_only}

        if self.args.use_mask:
            outputs['pred_mask'] = inputs['observed_mask'][:, -1:, :].repeat(1, self.args.pred_frames_num, 1)

        return outputs
