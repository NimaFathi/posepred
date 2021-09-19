import torch

from models.pv_lstm import PVLSTM
from models.zero_vel import ZeroVel


class Disentangled(torch.nn.Module):
    def __init__(self, args):
        super(Disentangled, self).__init__()
        self.args = args

        # global
        global_args = args
        global_args.keypoints_num = 1
        self.global_model = ZeroVel(global_args)

        # local
        local_args = args
        local_args.keypoints_num = args.keypoints_num - global_args.keypoints_num
        self.local_model = PVLSTM(local_args)

    def forward(self, inputs):
        outputs = []
        pose, vel = inputs[:2]

        # global
        global_pose = pose[..., : self.args.keypoint_dim]
        global_vel = vel[..., : self.args.keypoint_dim]
        global_inputs = [global_pose, global_vel]

        # local
        repeat = torch.ones(len(global_pose.shape), dtype=int)
        repeat[-1] = self.local_model.args.keypoints_num
        local_pose = pose[..., self.args.keypoint_dim:] - global_pose.repeat(tuple(repeat))
        local_vel = vel[..., self.args.keypoint_dim:] + global_vel.repeat(tuple(repeat))
        local_inputs = [local_pose, local_vel]

        if self.args.use_mask:
            mask = inputs[2]
            global_inputs.append(mask[..., :1])
            local_inputs.append(pose[..., 1:])

        # predict
        global_outputs = self.global_model(global_inputs)
        local_outputs = self.local_model(local_inputs)

        # merge local and global velocity
        global_vel_out = global_outputs[0]
        local_vel_out = local_outputs[0]
        repeat = torch.ones(len(global_vel_out.shape), dtype=int)
        repeat[-1] = self.local_model.args.keypoints_num
        outputs.append(torch.cat((global_vel_out, local_vel_out + global_vel_out.repeat(tuple(repeat))), dim=-1))

        if self.args.use_mask:
            outputs.append(torch.cat((global_outputs[-1], local_outputs[-1]), dim=-1))

        return tuple(outputs)
