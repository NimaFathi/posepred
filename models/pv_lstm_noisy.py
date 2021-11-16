import torch
import torch.nn as nn

from utils.others import pose_from_vel, get_dct_matrix


class PVLSTMNoisy(nn.Module):
    def __init__(self, args):
        super(PVLSTMNoisy, self).__init__()
        self.args = args
        input_size = output_size = int(args.keypoints_num * args.keypoint_dim)

        self.pose_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)
        self.vel_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)

        assert args.activation_type in ['hardtanh', 'none'], 'invalid activation_function.'
        self.vel_decoder = Decoder(args.pred_frames_num, input_size, output_size, args.hidden_size, args.n_layers,
                                   args.dropout_pose_dec, args.activation_type, args.hardtanh_limit, device=args.device)
        self.completion = Completion(input_size, output_size, args.hidden_size, args.n_layers, args.dropout_pose_dec,
                                     args.autoregressive, args.activation_type, args.hardtanh_limit, device=args.device)

        if self.args.use_dct:
            # observed_vel
            dct_obs_vel, idct_obs_vel = get_dct_matrix(self.args.obs_frames_num - 1)
            self.dct_obs_vel = torch.from_numpy(dct_obs_vel).float().to(self.args.device)
            self.idct_obs_vel = torch.from_numpy(idct_obs_vel).float().to(self.args.device)
            # observed_pose
            dct_obs_pose, _ = get_dct_matrix(self.args.obs_frames_num)
            self.dct_obs_pose = torch.from_numpy(dct_obs_pose).float().to(self.args.device)
            # future_vel
            _, idct_future_vel = get_dct_matrix(self.args.pred_frames_num)
            self.idct_future_vel = torch.from_numpy(idct_future_vel).float().to(self.args.device)

    def forward(self, inputs):
        pose = inputs['observed_pose']
        vel = pose[..., 1:, :] - pose[..., :-1, :]

        if 'observed_noise' not in inputs.keys():
            raise Exception("This model requires noise. set is_noisy to True")

        # make vel noisy
        bs, frames_n, features_n = vel.shape
        vel_ = vel.reshape(bs, frames_n, self.args.keypoints_num, self.args.keypoint_dim)
        vel_noise = torch.logical_or(inputs['observed_noise'][:, :-1, :], inputs['observed_noise'][:, 1:, :]).reshape(
            bs, frames_n, self.args.keypoints_num, 1)
        vel_noise = vel_noise.repeat(1, 1, 1, self.args.keypoint_dim)
        const = (torch.ones_like(vel_noise, dtype=torch.float) * self.args.noise_value)
        noisy_vel = torch.where(vel_noise == 1, const, vel_).reshape(bs, frames_n, -1)

        # make pose noisy
        bs, frames_n, features_n = pose.shape
        pose_ = pose.reshape(bs, frames_n, self.args.keypoints_num, self.args.keypoint_dim)
        pose_noise = inputs['observed_noise'].reshape(bs, frames_n, self.args.keypoints_num, 1)
        pose_noise = pose_noise.repeat(1, 1, 1, self.args.keypoint_dim)
        const = (torch.ones_like(pose_noise, dtype=torch.float) * self.args.noise_value)
        noisy_pose = torch.where(pose_noise == 1, const, pose_).reshape(bs, frames_n, -1)

        if self.args.use_dct:
            noisy_vel = torch.matmul(self.dct_obs_vel.unsqueeze(0), noisy_vel)
            noisy_pose = torch.matmul(self.dct_obs_pose.unsqueeze(0), noisy_pose)

        # vel_encoder
        (hidden_vel, cell_vel) = self.vel_encoder(noisy_vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        # pose_encoder
        (hidden_pose, cell_pose) = self.pose_encoder(noisy_pose.permute(1, 0, 2))
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)

        # hidden and cell for decoders
        hidden_dec = hidden_pose + hidden_vel
        cell_dec = cell_pose + cell_vel

        # vel_decoder
        pred_vel = self.vel_decoder(noisy_vel[:, -1, :], hidden_dec, cell_dec)
        if self.args.use_dct:
            pred_vel = torch.matmul(self.idct_future_vel.unsqueeze(0), pred_vel)
        pred_pose = pose_from_vel(pred_vel, noisy_pose[..., -1, :])

        # completion
        comp_vel = self.completion(noisy_vel, hidden_dec, cell_dec)
        if self.args.use_dct:
            comp_vel = torch.matmul(self.idct_obs_vel.unsqueeze(0), comp_vel)
        comp_pose = torch.clone(noisy_pose)
        for i in range(comp_vel.shape[-2]):
            comp_pose[..., i + 1, :] = comp_pose[..., i, :] + comp_vel[..., i, :]

        bs, frames_n, feat = comp_pose.shape
        comp_pose_noise_only = torch.where(pose_noise.view(bs, frames_n, feat) == 1, comp_pose, pose)

        outputs = {'pred_pose': pred_pose, 'pred_vel': pred_vel, 'comp_pose': comp_pose, 'comp_vel': comp_vel,
                   'comp_pose_noise_only': comp_pose_noise_only}

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


class Decoder(nn.Module):
    def __init__(self, outputs_num, input_size, output_size, hidden_size, n_layers, dropout, activation_type,
                 hardtanh_limit, device):
        super().__init__()
        self.device = device
        self.outputs_num = outputs_num
        self.dropout = nn.Dropout(dropout)
        lstms = [
            nn.LSTMCell(input_size=input_size if i == 0 else hidden_size, hidden_size=hidden_size) for
            i in range(n_layers)]
        self.lstms = nn.Sequential(*lstms)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=output_size)
        if activation_type == 'hardtanh':
            self.activation = nn.Hardtanh(min_val=-1 * hardtanh_limit, max_val=hardtanh_limit, inplace=False)
        elif activation_type == 'none':
            self.activation = None
        else:
            raise Exception("invalid activation_type.")

    def forward(self, inputs, hiddens, cells):
        dec_inputs = self.dropout(inputs)
        if len(hiddens.shape) < 3 or len(cells.shape) < 3:
            hiddens = torch.unsqueeze(hiddens, 0)
            cells = torch.unsqueeze(cells, 0)
        outputs = torch.tensor([], device=self.device)
        for j in range(self.outputs_num):
            for i, lstm in enumerate(self.lstms):
                if i == 0:
                    hiddens[i], cells[i] = lstm(dec_inputs, (hiddens.clone()[i], cells.clone()[i]))
                else:
                    hiddens[i], cells[i] = lstm(hiddens.clone()[i - 1], (hiddens.clone()[i], cells.clone()[i]))
            if self.activation is not None:
                output = self.activation(self.fc_out(hiddens.clone()[-1]))
            else:
                output = self.fc_out(hiddens.clone()[-1])
            dec_inputs = output.detach()
            outputs = torch.cat((outputs, output.unsqueeze(1)), 1)
        return outputs


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
