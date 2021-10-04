import torch
import torch.nn as nn
import random


class MixAndMatch(nn.Module):
    def __init__(self, args):
        super(MixAndMatch, self).__init__()
        self.args = args
        args.latent_dim = 32
        # the list containing the random indices sampled by "Sampling" operation
        self.sampled_indices = []
        self.complementary_indices = []
        input_size = output_size = int(args.keypoints_num * args.keypoint_dim)
        self.teacher_forcing_rate = 1
        self.count = 0

        # the encoder (note, it can be any neural network, e.g., GRU, Linear, ...)
        self.past_encoder = GRUEncoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)
        self.future_encoder = GRUEncoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)
        self.future_decoder = GRUDecoder(14, input_size, output_size, args.hidden_size, args.n_layers,
                                         args.dropout_pose_dec, 'hardtanh', args.hardtanh_limit
                                         )
        self.data_decoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.BatchNorm1d(args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU()
        )

        # layers to compute the data posterior
        self.mean = nn.Linear(args.hidden_size, args.latent_dim)
        self.std = nn.Linear(args.hidden_size, args.latent_dim)
        # layer to map the latent variable back to hidden size
        self.decode_latent = nn.Sequential(nn.Linear(args.latent_dim, args.hidden_size), nn.ReLU())

    def encode_past(self, condition):
        h = self.past_encoder(condition)
        h = h.squeeze(0)
        # sampled_indices are the ones that has been sampled by the "Sampling" operation
        # complementary_indices are the complementary set of indices that
        # has not been sampled in the "Sampling" operation.
        sampled_observed = h[:, self.sampled_indices]
        complementary_observed = h[:, self.complementary_indices]

        return sampled_observed, complementary_observed

    def encode_future(self, data):
        h = self.future_encoder(data)
        h = h.squeeze(0)
        # sampled_indices are the ones that has been sampled by the "Sampling" operation
        # complementary_indices are the complementary set of indices that
        # has not been sampled in the "Sampling" operation.
        sampled_data = h[:, self.sampled_indices]
        complementary_data = h[:, self.complementary_indices]

        return sampled_data, complementary_data

    def encode(self, sampled_data, complementary_condition):

        # Resample
        # creating a new vector (the result of conditioning the encoder)
        # that the total size is args.hidden_dim
        fusion = torch.zeros(sampled_data.shape[0], sampled_data.shape[1] + complementary_condition.shape[1]).to(self.args.device)
        fusion[:, self.sampled_indices] = sampled_data
        fusion[:, self.complementary_indices] = complementary_condition

        # compute the mean and standard deviation of the approximate posterior
        mu = self.mean(fusion)
        sigma = self.std(fusion)

        # reparameterization for sampling from the approximate posterior
        return self.reparameterize(mu, sigma), mu, sigma

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.args.device)
        return eps.mul(std).add_(mu)

    def decode(self, latent, sampled_condition):
        latent = self.decode_latent(latent)
        complementary_latent = latent[:, self.complementary_indices]

        # Resample
        fusion = torch.zeros(sampled_condition.shape[0], self.args.hidden_size).to(self.args.device)
        fusion[:, self.sampled_indices] = sampled_condition
        fusion[:, self.complementary_indices] = complementary_latent
        return self.data_decoder(fusion)

    def forward(self, inputs, observed_poses, future_poses, alpha):
        observed_poses = inputs['observed_pose']
        alpha = inputs['alpha']
        future_poses = inputs['future_pose']
        # The fist step is to perform "Sampling"
        # the parameter "alpha" is the pertubation rate. it is usually half of hidden_dim
        self.sampled_indices = list(random.sample(range(0, self.args.hidden_size), alpha))
        self.complementary_indices = [i for i in range(self.args.hidden_size) if i not in self.sampled_indices]
        # encode data and condition
        sampled_future, complementary_future = self.encode_future(future_poses.permute(1, 0, 2))
        sampled_past, complementary_past = self.encode_past(observed_poses.permute(1, 0, 2))
        # VAE encoder
        z, mu, sigma = self.encode(sampled_future, complementary_past)

        # VAE decoder
        decoded = self.decode(z, sampled_past)
        pred_poses = self.future_decoder(input=observed_poses[:, -1, :], future_poses=future_poses, hiddens=decoded, teacher_forcing_rate=self.teacher_forcing_rate)
        self.__update_teacher_forcing_rate()
        outputs = {'pred_pose': pred_poses, 'mu': mu, 'sigma': sigma}
        return decoded, mu, sigma

    def sample(self, obs, alpha, z=None):
        # The fist step is to perform "Sampling"
        # the parameter "alpha" is the perturbation rate. it is usually half of hidden_dim
        self.sampled_indices = list(random.sample(range(0, self.args.hidden_dim), alpha))
        self.complementary_indices = [i for i in range(self.args.hidden_dim) if i not in self.sampled_indices]

        # encode the condition
        sampled_past, complementary_past = self.encode_past(obs)

        # draw a sample from the prior distribution
        if z is None:
            z = torch.randn(obs.shape[0], self.args.latent_dim).normal_(0, 1).to(self.args.device)

        # VAE decoder
        generated = self.decode(z, sampled_past)

        return generated

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
