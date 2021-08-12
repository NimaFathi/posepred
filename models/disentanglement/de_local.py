import torch.nn as nn

from models.decoder import VelDecoder
from models.encoder import Encoder


class DisentangleLocal(nn.Module):
    def __init__(self, args):
        super(DisentangleLocal, self).__init__()
        self.args = args
        self.pose_encoder = Encoder(args=self.args, input_size=args.pose_input_size)
        self.vel_encoder = Encoder(args=self.args, input_size=args.pose_input_size)
        self.vel_decoder = VelDecoder(args=self.args, out_features=args.pose_out_features,
                                      input_size=args.pose_input_size)

    def forward(self, pose=None, vel=None):
        outputs = []

        (hidden_vel, cell_vel) = self.vel_encoder(vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        (hidden_pose, cell_pose) = self.pose_encoder(pose.permute(1, 0, 2))
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)

        VelDec_inp = vel[:, -1, :]

        hidden_dec = hidden_pose + hidden_vel
        cell_dec = cell_pose + cell_vel
        outputs.append(self.vel_decoder(VelDec_inp, hidden_dec, cell_dec))
        return outputs
