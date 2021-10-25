import torch
import torch.nn as nn


class PVLSTMComp(nn.Module):
    def __init__(self, args):
        super(PVLSTMComp, self).__init__()
        self.args = args
        input_size = output_size = int(args.keypoints_num * args.keypoint_dim)
        self.pose_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)
        self.vel_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)
        self.vel_decoder = Completion(input_size, output_size, args.hidden_size, args.n_layers, args.dropout_pose_dec,
                                      args.autoregressive, args.activation_type, args.hardtanh_limit, args.device)

    def forward(self, inputs):
        pose = inputs['observed_pose']
        vel = pose[..., 1:, :] - pose[..., :-1, :]

        bs, frames_n, features_n = vel.shape
        # make data noisy
        if 'observed_noise' in inputs.keys():
            noise = inputs['observed_noise'][:, 1:, :]
        else:
            raise Exception("This model requires noise. set is_noisy to True")

        vel = vel.reshape(bs, frames_n, self.args.keypoints_num, self.args.keypoint_dim)
        noise = noise.reshape(bs, frames_n, self.args.keypoints_num, 1).repeat(1, 1, 1, self.args.keypoint_dim)
        const = (torch.ones_like(noise, dtype=torch.float) * self.args.noise_value)
        vel = torch.where(noise == 1, const, vel).reshape(bs, frames_n, -1)

        (hidden_vel, cell_vel) = self.vel_encoder(vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        (hidden_pose, cell_pose) = self.pose_encoder(pose.permute(1, 0, 2))
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)

        vel_dec_input = vel[:, -1, :]
        hidden_dec = hidden_pose + hidden_vel
        cell_dec = cell_pose + cell_vel
        comp_vel = self.vel_decoder(vel_dec_input, hidden_dec, cell_dec)
        comp_pose = torch.clone(pose)
        for i in range(comp_vel.shape[-2]):
            comp_pose[..., i + 1, :] = comp_pose[..., i, :] + comp_vel[..., i, :]

        outputs = {'pred_pose': inputs['future_pose'], 'comp_pose': comp_pose, 'comp_vel': comp_vel}

        if self.args.use_mask:
            outputs['pred_mask'] = inputs['observed_mask'][:, -1:, :].repeat(1, self.args.pred_frames_num, 1)

        return outputs


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)

    def forward(self, inpput_):
        outputs, (hidden, cell) = self.lstm(inpput_)
        return hidden, cell


class Completion(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout, autoregressive, activation_type,
                 hardtanh_limit, device):
        super().__init__()
        self.device = device
        self.autoregressive = autoregressive
        self.dropout = nn.Dropout(dropout)
        lstms = [
            nn.LSTMCell(input_size=input_size if i == 0 else hidden_size, hidden_size=hidden_size) for
            i in range(n_layers)]
        self.lstms = nn.Sequential(*lstms)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=output_size)
        if activation_type == 'hardtanh':
            self.activation = nn.Hardtanh(min_val=-1 * hardtanh_limit, max_val=hardtanh_limit, inplace=False)
        elif activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'none':
            self.activation = None
        else:
            raise Exception("invalid activation_type.")

    def forward(self, inputs, hiddens, cells):
        frames_n = inputs.shape[-2]
        dec_inputs = self.dropout(inputs)
        if len(hiddens.shape) < 3 or len(cells.shape) < 3:
            hiddens = torch.unsqueeze(hiddens, 0)
            cells = torch.unsqueeze(cells, 0)
        outputs = torch.tensor([], device=self.device)

        output = dec_inputs[..., 0, :]
        for j in range(frames_n):
            dec_input = output.detach() if self.autoregressive else dec_inputs[..., j, :].detach()
            for i, lstm in enumerate(self.lstms):
                if i == 0:
                    hiddens[i], cells[i] = lstm(dec_input, (hiddens.clone()[i], cells.clone()[i]))
                else:
                    hiddens[i], cells[i] = lstm(hiddens.clone()[i - 1], (hiddens.clone()[i], cells.clone()[i]))
            if self.activation is not None:
                output = self.activation(self.fc_out(hiddens.clone()[-1]))
            else:
                output = self.fc_out(hiddens.clone()[-1])
            outputs = torch.cat((outputs, output.unsqueeze(1)), 1)
        return outputs
