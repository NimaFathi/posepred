import torch
import torch.nn as nn


class DeRPoF(nn.Module):
    def __init__(self, args):
        super(DeRPoF, self).__init__()
        self.args = args

        input_size = int(args.keypoints_num * args.keypoint_dim)
        net_g = LSTM_g(pose_dim=input_size, embedding_dim=args.embedding_dim, h_dim=args.hidden_dim, dropout=args.dropout)
        encoder = Encoder(pose_dim=input_size, h_dim=args.hidden_dim, latent_dim=args.latent_dim, dropout=args.dropout)
        decoder = Decoder(pose_dim=input_size, h_dim=args.hidden_dim, latent_dim=args.latent_dim, dropout=args.dropout)
        net_l = VAE(Encoder=encoder, Decoder=decoder)
        if torch.cuda.is_available():
            net_l.cuda()
            net_g.cuda()
        net_l.double()
        net_g.double()
        net_params = list(net_l.parameters()) + list(net_g.parameters())



        # global
        global_args = args
        global_args.keypoints_num = 1
        self.global_model = ZeroVel(global_args)

        # local
        local_args = args
        local_args.keypoints_num = args.keypoints_num - global_args.keypoints_num
        self.local_model = PVLSTM(local_args)

    def forward(self, inputs):
        outputs = []
        pose, vel = inputs[:2]

        # global
        global_pose = pose[..., : self.args.keypoint_dim]
        global_vel = vel[..., : self.args.keypoint_dim]
        global_inputs = [global_pose, global_vel]

        # local
        repeat = torch.ones(len(global_pose.shape), dtype=int)
        repeat[-1] = self.local_model.args.keypoints_num
        local_pose = pose[..., self.args.keypoint_dim:] - global_pose.repeat(tuple(repeat))
        local_vel = vel[..., self.args.keypoint_dim:] + global_vel.repeat(tuple(repeat))
        local_inputs = [local_pose, local_vel]

        if self.args.use_mask:
            mask = inputs[2]
            global_inputs.append(mask[..., :1])
            local_inputs.append(pose[..., 1:])

        # predict
        global_outputs = self.global_model(global_inputs)
        local_outputs = self.local_model(local_inputs)

        # merge local and global velocity
        global_vel_out = global_outputs[0]
        local_vel_out = local_outputs[0]
        repeat = torch.ones(len(global_vel_out.shape), dtype=int)
        repeat[-1] = self.local_model.args.keypoints_num
        outputs.append(torch.cat((global_vel_out, local_vel_out + global_vel_out.repeat(tuple(repeat))), dim=-1))

        if self.args.use_mask:
            outputs.append(torch.cat((global_outputs[-1], local_outputs[-1]), dim=-1))

        return tuple(outputs)


class LSTM_g(nn.Module):
    def __init__(self, pose_dim, embedding_dim=8, h_dim=16, num_layers=1, dropout=0.2):
        super(LSTM_g, self).__init__()
        self.doc = "LSTM_local_global"
        self.pose_dim = pose_dim
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.embedding_fn = nn.Sequential(nn.Linear(3, embedding_dim), nn.Tanh())
        self.encoder_g = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.decoder_g = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.hidden2g = nn.Sequential(nn.Linear(h_dim, 3))

    def forward(self, global_s=None, pred_len=14):
        seq_len, batch, l = global_s.shape
        state_tuple_g = (torch.zeros(self.num_layers, batch, self.h_dim, device='cpu', dtype=torch.float64),
                         torch.zeros(self.num_layers, batch, self.h_dim, device='cpu', dtype=torch.float64))

        global_s = global_s.contiguous()
        output_g, state_tuple_g = self.encoder_g(
            self.embedding_fn(global_s.view(-1, 3)).view(seq_len, batch, self.embedding_dim), state_tuple_g)

        pred_s_g = torch.tensor([], device='cpu')
        last_s_g = global_s[-1].unsqueeze(0)
        for _ in range(pred_len):
            output_g, state_tuple_g = self.decoder_g(
                self.embedding_fn(last_s_g.view(-1, 3)).view(1, batch, self.embedding_dim), state_tuple_g)
            curr_s_g = self.hidden2g(output_g.view(-1, self.h_dim))
            pred_s_g = torch.cat((pred_s_g, curr_s_g.unsqueeze(0)), dim=0)
            last_s_g = curr_s_g.unsqueeze(0)

        return pred_s_g


class Encoder(nn.Module):
    def __init__(self, pose_dim, h_dim=32, latent_dim=16, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()

        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(pose_dim, h_dim, num_layers, dropout=dropout)
        self.FC_mean = nn.Linear(h_dim, latent_dim)
        self.FC_var = nn.Linear(h_dim, latent_dim)

    def forward(self, obs_s=None):
        batch = obs_s.size(1)
        state_tuple = (torch.zeros(self.num_layers, batch, self.h_dim, device='cpu', dtype=torch.float64),
                       torch.zeros(self.num_layers, batch, self.h_dim, device='cpu', dtype=torch.float64))
        output, state_tuple = self.encoder(obs_s, state_tuple)
        out = output[-1]
        mean = self.FC_mean(out)
        log_var = self.FC_var(out)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, pose_dim, h_dim=32, latent_dim=16, num_layers=1, dropout=0.3):
        super(Decoder, self).__init__()
        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.decoder = nn.LSTM(pose_dim, h_dim, num_layers, dropout=dropout)
        self.FC = nn.Sequential(nn.Linear(latent_dim, h_dim))
        self.mlp = nn.Sequential(nn.Linear(h_dim, pose_dim))

    def forward(self, obs_s=None, latent=None, pred_len=14):
        batch = obs_s.size(1)
        decoder_c = torch.zeros(self.num_layers, batch, self.h_dim, device='cpu', dtype=torch.float64)
        last_s = obs_s[-1].unsqueeze(0)
        decoder_h = self.FC(latent).unsqueeze(0)
        decoder_h = decoder_h.repeat(self.num_layers, 1, 1)
        state_tuple = (decoder_h, decoder_c)

        preds_s = torch.tensor([], device='cpu')
        for _ in range(pred_len):
            output, state_tuple = self.decoder(last_s, state_tuple)
            curr_s = self.mlp(output.view(-1, self.h_dim))
            preds_s = torch.cat((preds_s, curr_s.unsqueeze(0)), dim=0)
            last_s = curr_s.unsqueeze(0)

        return preds_s


class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, obs_s=None):
        mean, log_var = self.Encoder(obs_s=obs_s)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        preds_s = self.Decoder(obs_s=obs_s, latent=z)

        return preds_s, mean, log_var


def vae_loss_function(x, x_hat, mean, log_var):
    # BCE_loss = nn.BCELoss()
    # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    assert x_hat.shape == x.shape
    reconstruction_loss = torch.mean(torch.norm(x - x_hat, dim=len(x.shape) - 1))
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + 0.01 * KLD
