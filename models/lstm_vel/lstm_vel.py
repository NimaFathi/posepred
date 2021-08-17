import torch.nn as nn

from models.decoder import VelDecoder
from models.encoder import Encoder


class LSTMVel(nn.Module):
    def __init__(self, args, pred_frames_num, dim, keypoints_num):
        super(LSTMVel, self).__init__()
        input_size = output_size = keypoints_num * dim
        self.pose_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_encoder)
        self.vel_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_encoder)
        self.vel_decoder = VelDecoder(pred_frames_num, input_size, output_size, args.hardtanh_limit, args.hidden_size,
                                      args.n_layers, args.dropout_pose_dec)

    def forward(self, pose=None, vel=None):
        outputs = []

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
        return tuple(outputs)
