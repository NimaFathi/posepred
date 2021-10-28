import torch
import torch.nn as nn
import math
import numpy as np
from utils.others import pose_from_vel
from scipy.fftpack import dct, idct
from utils.others import get_dct_matrix


class CompPredVelDCT(nn.Module):
    def __init__(self, args):
        super(CompPredVelDCT, self).__init__()
        self.args = args
        input_size = output_size = int(args.keypoints_num * args.keypoint_dim)

        self.vel_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)

        self.res_block1 = ResidualBlock(input_size=args.hidden_size, embedding_size=args.hidden_size)
        self.mean = nn.Linear(args.hidden_size, args.latent_dim)
        self.std = nn.Linear(args.hidden_size, args.latent_dim)

        self.decode_latent = nn.Sequential(nn.Linear(args.latent_dim, args.hidden_size), nn.ReLU())
        self.res_block2 = ResidualBlock(input_size=args.hidden_size, embedding_size=args.hidden_size)
        self.data_decoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.BatchNorm1d(args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU()
        )
        assert args.activation_type in ['hardtanh', 'sigmoid', 'none'], 'invalid activation_function.'
        self.vel_decoder = Decoder(args.pred_frames_num, input_size, output_size, args.hidden_size, args.n_layers,
                                   args.dropout_pose_dec, args.activation_type, args.hardtanh_limit, device=args.device)

        self.completion = Completion(input_size, output_size, args.hidden_size, args.n_layers, args.dropout_pose_dec,
                                     args.autoregressive, args.activation_type, args.hardtanh_limit, device=args.device)

    def forward(self, inputs):
        pose = inputs['observed_pose']
        print(pose[0][0])
        print(pose[0][1])
        exit()
        dct_pose = dct(pose.detach().cpu().numpy(), type=2, norm='ortho')
        vel = dct_pose[..., 1:, :] - dct_pose[..., :-1, :]
        bs, frames_n, features_n = vel.shape
        # dct_m, idct_m = get_dct_matrix(10 + self.args.pred_frames_num)
        print(pose.shape)
        # print(dct_m.shape)
        print(pose[0][0][:21])
        print(dct_pose[0][0][:21])
        exit()
        # make data noisy
        if 'observed_noise' in inputs.keys():
            noise = inputs['observed_noise'][:, 1:, :]
        else:
            raise Exception("This model requires noise. set is_noisy to True")

        vel = vel.reshape(bs, frames_n, self.args.keypoints_num, self.args.keypoint_dim)
        noise = noise.reshape(bs, frames_n, self.args.keypoints_num, 1).repeat(1, 1, 1, self.args.keypoint_dim)
        const = (torch.ones_like(noise, dtype=torch.float) * self.args.noise_value)
        noisy_vel = torch.where(noise == 1, const, vel).reshape(bs, frames_n, -1)

        # velocity encoder
        (hidden_vel, cell_vel) = self.vel_encoder(noisy_vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        # VAE encoder
        fusion1 = self.res_block1(hidden_vel, nn.ReLU())
        mean = self.mean(fusion1)
        std = self.std(fusion1)

        # VAE decoder
        latent = self.reparameterize(mean, std)
        fusion2 = self.decode_latent(latent)
        hidden_vel = self.res_block2(fusion2, nn.ReLU())

        # velocity decoder
        zeros = torch.zeros_like(cell_vel)
        pred_vel = self.vel_decoder(noisy_vel[..., -1, :], hidden_vel, zeros)
        pred_pose = pose_from_vel(pred_vel, pose[..., -1, :])

        # completion
        zeros = torch.zeros_like(cell_vel)
        comp_vel = self.completion(noisy_vel, hidden_vel, zeros)
        comp_pose = torch.clone(pose)
        for i in range(comp_vel.shape[-2]):
            comp_pose[..., i + 1, :] = comp_pose[..., i, :] + comp_vel[..., i, :]

        outputs = {'pred_pose': pred_pose, 'pred_vel': pred_vel, 'comp_pose': comp_pose, 'comp_vel': comp_vel,
                   'mean': mean, 'std': std, 'noise': noise}

        if self.args.use_mask:
            outputs['pred_mask'] = inputs['observed_mask'][:, -1:, :].repeat(1, self.args.pred_frames_num, 1)

        return outputs

    def reparameterize(self, mean, std):
        eps = torch.randn_like(mean).to(self.args.device)
        return eps.mul(torch.exp(0.5 * std)).add_(mean)


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
        elif activation_type == 'sigmiod':
            self.activation = nn.Sigmoid()
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
        elif activation_type == 'sigmiod':
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


class ResidualBlock(nn.Module):
    """ Residual Network that is then used for the VAE encoder and the VAE decoder. """

    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.shortcut = nn.Linear(input_size, embedding_size)
        self.deep1 = nn.Linear(input_size, embedding_size // 2)
        self.deep2 = nn.Linear(embedding_size // 2, embedding_size // 2)
        self.deep3 = nn.Linear(embedding_size // 2, embedding_size)

    def forward(self, input_tensor, activation=None):
        if activation is not None:
            shortcut = activation(self.shortcut(input_tensor))
            deep1 = activation(self.deep1(input_tensor))
            deep2 = activation(self.deep2(deep1))
            deep3 = activation(self.deep3(deep2))
        else:
            shortcut = self.shortcut(input_tensor)
            deep1 = self.deep1(input_tensor)
            deep2 = self.deep2(deep1)
            deep3 = self.deep3(deep2)

        output = shortcut + deep3

        return output

if __name__ == '__main__':
    pose1 = np.array([ 0.6900, -3.1700, 15.5600,  0.6900, -3.0500, 15.6000,  0.6900, -2.9400,
        15.6400,  0.6700, -2.9000, 15.6000,  0.5400, -2.8500, 15.6100,  0.4500,
        -2.6600, 15.7800,  0.3600, -2.4700, 15.9100,  0.7400, -2.9000, 15.6200,
         0.8300, -2.8700, 15.7000,  0.8900, -2.6200, 15.6900,  0.9400, -2.4300,
        15.5200,  0.6800, -2.7000, 15.7000,  0.6800, -2.5800, 15.7300,  0.6800,
        -2.5300, 15.7300,  0.6800, -2.4900, 15.7400,  0.6700, -2.4600, 15.7300,
         0.5800, -2.4000, 15.7000,  0.6900, -1.9800, 15.6100,  0.7000, -1.6000,
        15.7300,  0.7500, -2.4000, 15.7900,  0.7300, -2.0100, 15.9900,  0.5800,
        -1.9000, 16.3400]).reshape(1,-1)
    pose2 = np.array([ 0.7100, -3.1600, 15.4800,  0.7200, -3.0500, 15.5200,  0.7100, -2.9300,
        15.5600,  0.6900, -2.8900, 15.5200,  0.5600, -2.8400, 15.5400,  0.4800,
        -2.6300, 15.6700,  0.3900, -2.4200, 15.7900,  0.7600, -2.8900, 15.5400,
         0.8500, -2.8600, 15.6400,  0.9000, -2.6100, 15.6300,  0.9500, -2.4000,
        15.5000,  0.7100, -2.6900, 15.6200,  0.7000, -2.5700, 15.6400,  0.6900,
        -2.5200, 15.6600,  0.6900, -2.4800, 15.6700,  0.6900, -2.4600, 15.6500,
         0.6000, -2.4000, 15.6300,  0.6900, -1.9700, 15.5800,  0.7000, -1.6000,
        15.7300,  0.7700, -2.3900, 15.7200,  0.7700, -1.9700, 15.8500,  0.6300,
        -1.8900, 16.2000]).reshape(1, -1)
    dct1 = dct(pose1, type=2, norm='ortho')
    dct2 = dct(pose2, type=2, norm='ortho')
    # print(pose1.shape)
    dct_vel = dct2 - dct1
    vel = pose2 - pose1
    dct_vel_vel = dct(vel, norm='ortho')
    print(pose1 + idct(dct_vel, norm='ortho'))
    print(pose1 + idct(dct_vel_vel, norm='ortho'))
    print(pose2)