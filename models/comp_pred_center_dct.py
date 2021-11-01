import torch
import torch.nn as nn

from utils.others import get_dct_matrix


class CompPredCenterDCT(nn.Module):
    def __init__(self, args):
        super(CompPredCenterDCT, self).__init__()
        self.args = args
        input_size = output_size = int(args.keypoints_num * args.keypoint_dim)

        self.pose_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)

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

        self.pose_decoder = Decoder(args.pred_frames_num, input_size, output_size, args.hidden_size, args.n_layers,
                                    args.dropout_pose_dec, args.activation_type, args.hardtanh_limit, args.device)

        self.completion = Completion(input_size, output_size, args.hidden_size, args.n_layers, args.dropout_pose_dec,
                                     args.autoregressive, args.activation_type, args.hardtanh_limit, args.device)

    def forward(self, inputs):
        pose = inputs['observed_pose']
        torch.set_printoptions(threshold=10_000)

        bs, obs_frames_n, features_n = pose.shape
        first_frame = pose[:, 0:1, :]
        pose = pose - first_frame.repeat(1, obs_frames_n, 1)
        # make data noisy
        if 'observed_noise' in inputs.keys():
            noise = inputs['observed_noise']
        else:
            raise Exception("This model requires noise. set is_noisy to True")
        pose = pose.reshape(bs, obs_frames_n, self.args.keypoints_num, self.args.keypoint_dim)
        noise = noise.reshape(bs, obs_frames_n, self.args.keypoints_num, 1).repeat(1, 1, 1, self.args.keypoint_dim)
        const = (torch.ones_like(noise, dtype=torch.float) * self.args.noise_value)
        noisy_pose = torch.where(noise == 1, const, pose).reshape(bs, obs_frames_n, -1)
        dct_m, idct_m = get_dct_matrix(obs_frames_n)
        _, idct_p = get_dct_matrix(self.args.pred_frames_num)
        dct_m = torch.from_numpy(dct_m).float().to(self.args.device)
        idct_m = torch.from_numpy(idct_m).float().to(self.args.device)
        idct_p = torch.from_numpy(idct_p).float().to(self.args.device)
        noisy_dct_pose = torch.matmul(dct_m.unsqueeze(0), noisy_pose)
        # velocity encoder
        (hidden_p, cell_p) = self.pose_encoder(noisy_dct_pose.permute(1, 0, 2))
        hidden_p = hidden_p.squeeze(0)
        cell_p = cell_p.squeeze(0)

        # VAE encoder
        fusion1 = self.res_block1(hidden_p, nn.ReLU())
        mean = self.mean(fusion1)
        std = self.std(fusion1)

        # VAE decoder
        latent = self.reparameterize(mean, std)
        fusion2 = self.decode_latent(latent)
        hidden_p = self.res_block2(fusion2, nn.ReLU())

        # velocity decoder
        zeros = torch.zeros_like(cell_p)
        pred_dct_pose_center = self.pose_decoder(noisy_dct_pose[..., -1, :], hidden_p, zeros)
        # print(pred_dct_pose_center.shape)
        # print(idct_m.unsqueeze(0).shape)
        # exit()
        pred_pose_center = torch.matmul(idct_p.unsqueeze(0), pred_dct_pose_center)
        # pred_pose_center = torch.tensor(idct(pred_dct_pose_center.detach().cpu().numpy(), norm='ortho', axis=1)).to(self.args.device)
        pred_pose = pred_pose_center + first_frame.repeat(1, self.args.pred_frames_num, 1)

        # completion
        zeros = torch.zeros_like(cell_p)
        comp_dct_pose_center = self.completion(noisy_dct_pose, hidden_p, zeros)
        comp_pose_center = torch.matmul(idct_m.unsqueeze(0), comp_dct_pose_center)
        # comp_pose_center = torch.tensor(idct(comp_dct_pose_center.detach().cpu().numpy(), norm='ortho', axis=1)).to(self.args.device)
        comp_pose = comp_pose_center + first_frame.repeat(1, obs_frames_n, 1)

        outputs = {'pred_pose': pred_pose, 'pred_pose_center': pred_pose_center, 'comp_pose': comp_pose,
                   'comp_pose_center': comp_pose_center, 'mean': mean, 'std': std, 'noise': noise}

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
        elif activation_type == 'sigmoid':
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