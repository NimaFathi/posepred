import math

import torch
import torch.nn.functional as F
from torch import nn

from models.st_transformer.data_proc import Preprocess, Postprocess


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class diff_CSDI(nn.Module):
    def __init__(self, args, inputdim, side_dim):
        super().__init__()
        self.args = args
        self.channels = args.diff_channels  # config["channels"]

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,  # config["side_dim"],
                    channels=self.channels,
                    nheads=args.diff_nheads  # config["nheads"],
                )
                for _ in range(args.diff_layers)  # for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, nheads):
        super().__init__()
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        y = x
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class CSDI_base(nn.Module):
    # def __init__(self, target_dim, config, device):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.target_dim = args.keypoint_dim * args.n_major_joints

        self.emb_time_dim = args.model_timeemb
        self.emb_feature_dim = args.model_featureemb
        self.is_unconditional = args.model_is_unconditional

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(args, input_dim, self.emb_total_dim)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def forward(self, batch):
        (
            observed_data,
            observed_tp,
            cond_mask
        ) = self.preprocess_data(batch)

        side_info = self.get_side_info(observed_tp, cond_mask)

        B, K, L = observed_data.shape
        noisy_data = torch.zeros_like(observed_data).to(self.device)

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info)  # (B,K,L)
        return self.postprocess_data(batch, predicted)


# class CSDI_H36M(CSDI_base):
#     def __init__(self, args):
#         super(CSDI_H36M, self).__init__(args)
#         self.Lo = args.obs_frames_num
#         self.Lp = args.pred_frames_num
#
#         TDUL_T = self.Lp if args.loss.time_aware else 1
#         TDUL_J = 28 if args.loss.joint_aware else 1
#
#         if args.loss.action_aware:
#             self.sigma = nn.Embedding(15, TDUL_T * TDUL_J)
#             self.sigma.weight = nn.Parameter(torch.ones(15, TDUL_T * TDUL_J, requires_grad=True) * 3.5)
#         else:
#             self.sigma = nn.Parameter(torch.ones(TDUL_T, TDUL_J, requires_grad=True) * 3.5)
#
#         self.preprocess = Preprocess(args).to(args.device)
#         self.postprocess = Postprocess(args).to(args.device)
#
#         for p in self.preprocess.parameters():
#             p.requires_grad = False
#
#         for p in self.postprocess.parameters():
#             p.requires_grad = False
#
#
#     def preprocess_data(self, batch):
#         observed_data = batch["observed_pose"].to(self.device)
#         observed_data = self.preprocess(observed_data)
#
#         B, L, K = observed_data.shape
#         Lp = self.args.pred_frames_num
#
#         observed_data = observed_data.permute(0, 2, 1)  # B, K, L
#
#         observed_data = torch.cat([
#             observed_data, torch.zeros([B, K, Lp]).to(self.device)
#         ], dim=-1)
#
#         observed_tp = torch.arange(self.Lo + self.Lp).unsqueeze(0).expand(B, -1).to(self.device)
#         cond_mask = torch.zeros_like(observed_data).to(self.device)
#         cond_mask[:, :, :L] = 1
#
#         return (
#             observed_data,
#             observed_tp,
#             cond_mask
#         )
#
#     def postprocess_data(self, batch, predicted):
#         predicted = predicted[:, :, self.Lo:]
#         predicted = predicted.permute(0, 2, 1)
#         return {
#             'pred_pose': self.postprocess(batch['observed_pose'], predicted),  # B, T, 96
#             'sigma': self.sigma
#         }

class CSDI_H36M(CSDI_base):
    def __init__(self, args):
        super(CSDI_H36M, self).__init__(args)
        self.Lo = args.obs_frames_num
        self.Lp = args.pred_frames_num

        assert args.loss.time_aware and not args.loss.joint_aware and not args.loss.action_aware, 'in this implementation, only time are supported!'

        self.p1 = nn.Parameter(torch.ones(1, requires_grad=True) * 3.5)
        self.p2 = nn.Parameter(torch.ones(1, requires_grad=True) * 0.0)
        self.p3 = nn.Parameter(torch.ones(1, requires_grad=True) * 1.0)
        self.p4 = nn.Parameter(torch.ones(1, requires_grad=True) * 0.0)
        self.p5 = nn.Parameter(torch.ones(1, requires_grad=True) * 0.0)

        self.preprocess = Preprocess(args).to(args.device)
        self.postprocess = Postprocess(args).to(args.device)

        for p in self.preprocess.parameters():
            p.requires_grad = False

        for p in self.postprocess.parameters():
            p.requires_grad = False

    def preprocess_data(self, batch):
        observed_data = batch["observed_pose"].to(self.device)
        observed_data = self.preprocess(observed_data)

        B, L, K = observed_data.shape
        Lp = self.args.pred_frames_num

        observed_data = observed_data.permute(0, 2, 1)  # B, K, L

        observed_data = torch.cat([
            observed_data, torch.zeros([B, K, Lp]).to(self.device)
        ], dim=-1)

        observed_tp = torch.arange(self.Lo + self.Lp).unsqueeze(0).expand(B, -1).to(self.device)
        cond_mask = torch.zeros_like(observed_data).to(self.device)
        cond_mask[:, :, :L] = 1

        return (
            observed_data,
            observed_tp,
            cond_mask
        )

    def calc_sigma(self):
        x = torch.arange(self.Lp).to(self.args.device)
        c = 2 * self.p3 * self.p5 / torch.abs(self.p3 + self.p5)
        f = 1 / (1 + torch.exp(-c * (self.p4 - x)))
        g = torch.exp(self.p3 * (self.p4 - x))
        h = torch.exp(self.p5 * (self.p4 - x))
        sigma = self.p1 + (self.p2 / (1 + f * g + (1 - f) * h))
        return sigma.unsqueeze(1)

    def postprocess_data(self, batch, predicted):
        predicted = predicted[:, :, self.Lo:]
        predicted = predicted.permute(0, 2, 1)

        return {
            'pred_pose': self.postprocess(batch['observed_pose'], predicted),  # B, T, 96
            'sigma': self.calc_sigma()
        }
