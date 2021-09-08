import torch.nn as nn

from models.decoder import Decoder
from models.encoder import Encoder


class PVLSTM(nn.Module):
    def __init__(self, args):
        super(PVLSTM, self).__init__()
        self.args = args
        input_size = output_size = int(args.keypoints_num * args.keypoint_dim)
        self.pose_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)
        self.vel_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)
        self.vel_decoder = Decoder(args.pred_frames_num, input_size, output_size, args.hidden_size, args.n_layers,
                                   args.dropout_pose_dec, 'hardtanh', args.hardtanh_limit)

        if self.args.use_mask:
            self.mask_encoder = Encoder(args.keypoints_num, args.hidden_size, args.n_layers, args.dropout_encoder)
            self.mask_decoder = Decoder(args.pred_frames_num, args.keypoints_num, args.keypoints_num, args.hidden_size,
                                        args.n_layers, args.dropout_mask_dec, 'sigmoid')

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

        return tuple(outputs)
