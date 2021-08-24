import torch

from models.lstm_vel import LSTMVel
from models.zero_velocity import ZeroVelocity


class Disentangle1(torch.nn.Module):
    def __init__(self, args):
        super(Disentangle1, self).__init__()
        self.args = args

        # global
        global_args = args
        global_args.keypoints_num = 1
        self.global_model = ZeroVelocity(global_args).to(torch.device('cuda'))

        # local
        local_args = args
        local_args.keypoints_num = args.keypoints_num - global_args.keypoints_num
        self.local_model = LSTMVel(local_args).to(torch.device('cuda'))

    def forward(self, inputs):
        outputs = []
        pose, vel = inputs[:2]

        # (global_pose, global_vel)
        global_inputs = [pose[..., : self.args.keypoint_dim], vel[..., : self.args.keypoint_dim]]

        # (local_pose, local_vel)
        local_inputs = [pose[..., self.args.keypoint_dim:], vel[..., self.args.keypoint_dim:]]

        if self.args.use_mask:
            mask = inputs[2]
            global_inputs.append(mask[..., :1])
            local_inputs.append(pose[..., 1:])

        global_outputs = self.global_model(global_inputs)
        local_outputs = self.local_model(local_inputs)

        # merge local and global outputs
        global_vel = global_outputs[0]
        local_vel = local_outputs[0]
        repeat = torch.ones(len(global_vel.shape), dtype=int)
        repeat[-1] = self.local_model.args.keypoints_num
        outputs.append(torch.cat((global_vel, local_vel + global_vel.repeat(tuple(repeat))), dim=-1))

        return tuple(outputs)






