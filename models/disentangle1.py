import torch

#
# from models.lstm_vel import LSTMVel
# from models.zero_velocity import ZeroVelocity
#
#
# class Disentangle1(torch.nn.Module):
#     def __init__(self, args):
#         super(Disentangle1, self).__init__()
#         self.args = args
#
#         global_args = args
#         global_args.keypoints_num = 1
#         self.global_model = ZeroVelocity(global_args).to(torch.device('cuda'))
#
#         local_args = args
#         local_args.keypoints_num = args.keypoints_num - global_args.keypoints_num
#         self.local_model = LSTMVel(local_args).to(torch.device('cuda'))
#
#         self.input_size = self.output_size = int(args.keypoints_num * args.keypoint_dim)
#
#     def forward(self, inputs):
#         outputs = []
#         pose, vel = inputs[:2]
#         # [global_pose, global_vel]
#         global_inputs = [pose[..., : self.args.keypoint_dim], vel[..., : self.args.keypoint_dim]]
#         # [local_pose, local_vel]
#         local_inputs = [pose[..., self.args.keypoint_dim:], vel[..., self.args.keypoint_dim:]]
#
#         if self.args.use_mask:
#             mask = inputs[2]
#             global_inputs.append(mask[..., :1])
#             local_inputs.append(pose[..., 1:])
#
#         global_outputs = self.global_model(global_inputs)
#         local_inputs = self.local_model(local_inputs)
#
#         for i, global_output in enumerate(global_inputs):
#             local_output = local_inputs[i]
#
#             local_output[..., ]
#
#
#
#
#         return tuple(outputs)


local_ = torch.ones(32, 16, 12)


global_ = torch.ones(32, 16, 3)

global_r = global_.repeat(1, 1, 4)

print(global_.shape)
print(local_.shape)

print(global_r.shape)
print((local_ + global_r).shape)
res = torch.cat((global_, local_ + global_r), dim=-1)
print(res.shape)

# print(res)
