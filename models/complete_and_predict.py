import torch
import torch.nn as nn

from utils.others import pose_from_vel


class CompleteAndPredict(nn.Module):
    def __init__(self, args):
        super(CompleteAndPredict, self).__init__()
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

        self.vel_decoder = Decoder(args.pred_frames_num, input_size, output_size, args.hidden_size, args.n_layers,
                                   args.dropout_pose_dec, 'hardtanh', args.hardtanh_limit)

        self.completion = Completion(input_size, output_size, args.hidden_size, args.n_layers, args.dropout_pose_dec,
                                     'hardtanh', args.hardtanh_limit)

    def forward(self, inputs):
        pose = inputs['observed_pose']
        vel = pose[..., 1:, :] - pose[..., :-1, :]
        mask = inputs['observed_mask'][..., 1:, :]

        # make data noisy
        bs, frames_n, features_n = vel.shape
        vel = vel.reshape(bs, frames_n, self.args.keypoints_num, self.args.keypoint_dim)
        mask = mask.repeat(1, 1, 1, 3)
        const = torch.zeros_like(mask) * (-100)
        noisy_vel = torch.where(mask == 1, const, vel).reshape(bs, frames_n, -1)

        # velocity encoder
        (hidden_vel, cell_vel) = self.vel_encoder(noisy_vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        # VAE encoder
        fusion1 = self.res_block1(hidden_vel, nn.ReLU())
        latent = self.reparameterize(self.mean(fusion1), self.std(fusion1))

        # VAE decoder
        fusion2 = self.decode_latent(latent)
        hidden_vel2 = self.res_block2(fusion2, nn.ReLU())

        # velocity decoder
        vel_dec_input = noisy_vel[..., -1, :]
        pred_vel = self.vel_decoder(vel_dec_input, hidden_vel2, cell_vel)
        pred_pose = pose_from_vel(pred_vel, pose[..., -1, :])

        # completion
        complited_vel = self.completion(noisy_vel, hidden_vel2, cell_vel)

        outputs = {'pred_pose': pred_pose, 'pred_vel': pred_vel, 'completed_vel': complited_vel}

        return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.args.device)
        return eps.mul(std).add_(mu)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)

    def forward(self, inpput_):
        outputs, (hidden, cell) = self.lstm(inpput_)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, outputs_num, input_size, output_size, hidden_size, n_layers, dropout, activation_type,
                 hardtanh_limit=None):
        super().__init__()
        self.outputs_num = outputs_num
        self.dropout = nn.Dropout(dropout)
        lstms = [
            nn.LSTMCell(input_size=input_size if i == 0 else hidden_size, hidden_size=hidden_size) for
            i in range(n_layers)]
        self.lstms = nn.Sequential(*lstms)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=output_size)
        if activation_type == 'hardtanh':
            self.activation = nn.Hardtanh(min_val=-1 * hardtanh_limit, max_val=hardtanh_limit, inplace=False)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, inputs, hiddens, cells):
        dec_inputs = self.dropout(inputs)
        if len(hiddens.shape) < 3 or len(cells.shape) < 3:
            hiddens = torch.unsqueeze(hiddens, 0)
            cells = torch.unsqueeze(cells, 0)
        outputs = torch.tensor([], device=self.args.device)
        for j in range(self.outputs_num):
            for i, lstm in enumerate(self.lstms):
                if i == 0:
                    hiddens[i], cells[i] = lstm(dec_inputs, (hiddens.clone()[i], cells.clone()[i]))
                else:
                    hiddens[i], cells[i] = lstm(hiddens.clone()[i - 1], (hiddens.clone()[i], cells.clone()[i]))
            output = self.activation(self.fc_out(hiddens.clone()[-1]))
            dec_inputs = output.detach()
            outputs = torch.cat((outputs, output.unsqueeze(1)), 1)
        return outputs


class Completion(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout, activation_type,
                 hardtanh_limit=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        lstms = [
            nn.LSTMCell(input_size=input_size if i == 0 else hidden_size, hidden_size=hidden_size) for
            i in range(n_layers)]
        self.lstms = nn.Sequential(*lstms)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=output_size)
        if activation_type == 'hardtanh':
            self.activation = nn.Hardtanh(min_val=-1 * hardtanh_limit, max_val=hardtanh_limit, inplace=False)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, inputs, hiddens, cells):
        frames_n = inputs.shape[-2]
        dec_inputs = self.dropout(inputs)
        if len(hiddens.shape) < 3 or len(cells.shape) < 3:
            hiddens = torch.unsqueeze(hiddens, 0)
            cells = torch.unsqueeze(cells, 0)
        outputs = torch.tensor([], device=self.args.device)
        for j in range(frames_n):
            for i, lstm in enumerate(self.lstms):
                if i == 0:
                    hiddens[i], cells[i] = lstm(dec_inputs[..., j, :], (hiddens.clone()[i], cells.clone()[i]))
                else:
                    hiddens[i], cells[i] = lstm(hiddens.clone()[i - 1], (hiddens.clone()[i], cells.clone()[i]))
            output = self.activation(self.fc_out(hiddens.clone()[-1]))
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
