import random

import torch
import torch.nn as nn

from utils.others import qeuler, qfix


class MixAndMatch(nn.Module):
    def __init__(self, args):
        super(MixAndMatch, self).__init__()
        self.args = args
        if self.args.alpha is None:
            self.alpha = self.args.hidden_size // 2
        else:
            assert self.args.alpha <= self.args.hidden_size, 'alpha should be smaller than hidden_size'
            self.alpha = self.args.alpha
        # the list containing the random indices sampled by "Sampling" operation
        self.sampled_indices = []
        self.complementary_indices = []
        input_size = output_size = int(self.args.keypoints_num * 4)  # 4 for quaternion display
        self.teacher_forcing_rate = self.args.teacher_forcing_rate
        self.count = 0

        # the encoder (note, it can be any neural network, e.g., GRU, Linear, ...)
        self.past_encoder = GRUEncoder(input_size, self.args.hidden_size, self.args.n_layers, self.args.dropout_enc)
        self.future_encoder = GRUEncoder(input_size, self.args.hidden_size, self.args.n_layers, self.args.dropout_enc)
        self.future_decoder = GRUDecoder(self.args.pred_frames_num, input_size, output_size, self.args.hidden_size,
                                         self.args.n_layers,
                                         self.args.dropout_pose_dec, 'hardtanh', self.args.hardtanh_limit
                                         )
        self.data_decoder = nn.Sequential(
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.ReLU()
        )
        self.res_block1 = ResidualBlock(input_size=self.args.hidden_size, embedding_size=self.args.hidden_size)
        self.res_block2 = ResidualBlock(input_size=self.args.hidden_size, embedding_size=self.alpha)
        # layers to compute the data posterior
        self.mean = nn.Linear(self.args.hidden_size, self.args.latent_dim)
        self.std = nn.Linear(self.args.hidden_size, self.args.latent_dim)
        # layer to map the latent variable back to hidden size
        self.decode_latent = nn.Sequential(nn.Linear(self.args.latent_dim, self.args.hidden_size), nn.ReLU())

    def encode_past(self, condition):
        h = self.past_encoder(condition)
        h = h.squeeze(0)
        # sampled_indices are the ones that has been sampled by the "Sampling" operation
        # complementary_indices are the complementary set of indices that
        # has not been sampled in the "Sampling" operation.
        sampled_observed = h[:, self.sampled_indices]
        complementary_observed = h[:, self.complementary_indices]

        return h, sampled_observed, complementary_observed

    def encode_future(self, data):
        h = self.future_encoder(data)
        h = h.squeeze(0)
        # sampled_indices are the ones that has been sampled by the "Sampling" operation
        # complementary_indices are the complementary set of indices that
        # has not been sampled in the "Sampling" operation.
        sampled_data = h[:, self.sampled_indices]
        complementary_data = h[:, self.complementary_indices]

        return sampled_data, complementary_data

    def encode(self, sampled_past, complementary_future):

        # Resample
        # creating a new vector (the result of conditioning the encoder)
        # that the total size is args.hidden_dim
        fusion = torch.zeros(sampled_past.shape[0], sampled_past.shape[1] + complementary_future.shape[1]).to(
            self.args.device)
        fusion[:, self.sampled_indices] = sampled_past
        fusion[:, self.complementary_indices] = complementary_future
        fusion = self.res_block1(fusion, nn.ReLU())
        # compute the mean and standard deviation of the approximate posterior
        mu = self.mean(fusion)
        sigma = self.std(fusion)

        # reparameterization for sampling from the approximate posterior
        return self.reparameterize(mu, sigma), mu, sigma

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.args.device)
        return eps.mul(std).add_(mu)

    def decode(self, latent, sampled_past, complementary_past):
        latent = self.decode_latent(latent)
        complementary_latent = latent[:, self.complementary_indices]

        # Resample
        fusion1 = torch.zeros(sampled_past.shape[0], self.args.hidden_size).to(self.args.device)
        fusion1[:, self.sampled_indices] = sampled_past
        fusion1[:, self.complementary_indices] = complementary_latent
        h_zp = self.res_block2(fusion1, nn.ReLU())
        fusion2 = torch.zeros(sampled_past.shape[0], self.args.hidden_size).to(self.args.device)
        fusion2[:, self.sampled_indices] = h_zp
        fusion2[:, self.complementary_indices] = complementary_past
        return self.data_decoder(fusion2)

    def forward(self, inputs):
        assert {'observed_quaternion_pose', 'future_quaternion_pose'}.issubset(
            set(inputs.keys())), 'data should be in quaternion form'
        obs_q_poses = inputs['observed_quaternion_pose']
        future_q_poses = inputs['future_quaternion_pose']
        # The fist step is to perform "Sampling"
        # the parameter "alpha" is the pertubation rate. it is usually half of hidden_dim
        self.sampled_indices = list(random.sample(range(0, self.args.hidden_size), self.alpha))
        self.complementary_indices = [i for i in range(self.args.hidden_size) if i not in self.sampled_indices]
        # encode data and condition
        sampled_future, complementary_future = self.encode_future(future_q_poses.permute(1, 0, 2))
        hidden_pose, sampled_past, complementary_past = self.encode_past(obs_q_poses.permute(1, 0, 2))
        # VAE encoder
        z, mu, sigma = self.encode(sampled_past, complementary_future)

        # VAE decoder
        decoded = self.decode(z, sampled_past, complementary_past)
        pred_q_poses = self.future_decoder(input=obs_q_poses[:, -1, :], future_poses=future_q_poses, hiddens=decoded,
                                           teacher_forcing_rate=self.teacher_forcing_rate)
        self.__update_teacher_forcing_rate()
        quaternion_poses = pred_q_poses.clone().view(*pred_q_poses.shape[:-1], -1, 4)
        q_poses = quaternion_poses.cpu().detach().numpy()
        for i, q in enumerate(q_poses):
            q_poses[i] = qfix(q)
        quaternion_poses = torch.from_numpy(q_poses).to(self.args.device)
        pred_poses = qeuler(q=quaternion_poses, order='xyz')
        pred_poses = pred_poses.view(*pred_poses.shape[:-2], pred_poses.shape[-2] * pred_poses.shape[-1])
        outputs = {'pred_pose': pred_poses, 'pred_q_pose': pred_q_poses, 'mu': mu, 'sigma': sigma}
        return outputs

    # def sample(self, obs, alpha, z=None):
    #     # The fist step is to perform "Sampling"
    #     # the parameter "alpha" is the perturbation rate. it is usually half of hidden_dim
    #     self.sampled_indices = list(random.sample(range(0, self.args.hidden_dim), alpha))
    #     self.complementary_indices = [i for i in range(self.args.hidden_dim) if i not in self.sampled_indices]
    #
    #     # encode the condition
    #     sampled_past, complementary_past = self.encode_past(obs)
    #
    #     # draw a sample from the prior distribution
    #     if z is None:
    #         z = torch.randn(obs.shape[0], self.args.latent_dim).normal_(0, 1).to(self.args.device)
    #
    #     # VAE decoder
    #     generated = self.decode(z, sampled_past)
    #
    #     return generated

    def __update_teacher_forcing_rate(self):
        self.count += 1
        if self.count % 10 == 0:
            self.teacher_forcing_rate *= 0.96


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)

    def forward(self, inpput_):
        outputs, hidden = self.gru(inpput_)
        return hidden


class GRUDecoder(nn.Module):
    def __init__(self, outputs_num, input_size, output_size, hidden_size, n_layers, dropout, activation_type,
                 hardtanh_limit=None):
        super().__init__()
        self.outputs_num = outputs_num
        self.dropout = nn.Dropout(dropout)
        grus = [
            nn.GRUCell(input_size=input_size if i == 0 else hidden_size, hidden_size=hidden_size) for
            i in range(n_layers)]
        self.grus = nn.Sequential(*grus)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=output_size)
        if activation_type == 'hardtanh':
            self.activation = nn.Hardtanh(min_val=-1 * hardtanh_limit, max_val=hardtanh_limit, inplace=False)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, input, future_poses, hiddens, teacher_forcing_rate):
        dec_inputs = self.dropout(input)
        if len(hiddens.shape) < 3:
            hiddens = torch.unsqueeze(hiddens, 0)
        device = 'cuda' if input.is_cuda else 'cpu'
        outputs = torch.tensor([], device=device)
        for j in range(self.outputs_num):
            for i, gru in enumerate(self.grus):
                if i == 0:
                    hiddens[i] = gru(dec_inputs, hiddens.clone()[i])
                else:
                    hiddens[i] = gru(hiddens.clone()[i - 1], hiddens.clone()[i])
            if random.random() < teacher_forcing_rate:
                output = future_poses[:, j]
                output.requires_grad = True
            else:

                output = self.activation(self.fc_out(hiddens.clone()[-1]))
                dec_inputs = output.detach()
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
