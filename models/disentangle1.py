import torch

from models.lstm_vel import LSTMVel
from models.zero_velocity import ZeroVelocity


class Disentangle1(torch.nn.Module):
    def __init__(self, args):
        super(Disentangle1, self).__init__()
        self.args = args

        global_args = args
        global_args.keypoints_num = 1
        self.global_model = ZeroVelocity(global_args).to(torch.device('cuda'))

        local_args = args
        local_args.keypoints_num = args.keypoints_num - global_args.keypoints_num
        self.local_model = LSTMVel(local_args).to(torch.device('cuda'))

        self.input_size = self.output_size = int(args.keypoints_num * args.keypoint_dim)

    def forward(self, inputs):

        outputs = []
        pose, vel = inputs[:2]

        (hidden_vel, cell_vel) = self.vel_encoder(vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        (hidden_pose, cell_pose) = self.pose_encoder(pose.permute(1, 0, 2))
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)

        vel_dec_input = vel[:, -1, :]
        hidden_dec = hidden_pose + hidden_vel
        cell_dec = cell_pose + cell_vel
        outputs.append(self.vel_decoder(vel_dec_input, hidden_dec, cell_dec))

        if self.args.use_mask:
            mask = inputs[2]
            (hidden_mask, cell_mask) = self.mask_encoder(mask.permute(1, 0, 2))
            hidden_mask = hidden_mask.squeeze(0)
            cell_mask = cell_mask.squeeze(0)

            mask_dec_input = mask[:, -1, :]
            outputs.append(self.mask_decoder(mask_dec_input, hidden_mask, cell_mask))


        batch_size = inputs[0].shape[0]
        outputs.append(torch.zeros(batch_size, self.args.pred_frames_num, self.output_size))
        if self.args.use_mask:
            mask = inputs[2]
            last_frame = mask[..., -1, :].unsqueeze(-2)
            outputs.append(last_frame.repeat([1 for _ in range(len(mask.shape[:-2]))] + [self.args.pred_frames_num, 1]))

        return tuple(outputs)
