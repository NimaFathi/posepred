from turtle import forward
import torch
import torch.nn as nn
import sys
import numpy as np




def joint_to_index(x):
    return np.concatenate((x * 3, x * 3 + 1, x * 3 + 2))


dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

index_to_ignore = np.array([16, 20, 23, 24, 28, 31])
index_to_ignore = joint_to_index(index_to_ignore)

index_to_equal = np.array([13, 19, 22, 13, 27, 30])
index_to_equal = joint_to_index(index_to_equal)

index_to_copy = np.array([0, 1, 6, 11])
index_to_copy = joint_to_index(index_to_copy)


class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()

    def forward(self, observed_pose):
        return observed_pose[:, dim_used]


class Postprocess(nn.Module):
    def __init__(self):
        super(Postprocess, self).__init__()

    def forward(self, input_poses, sample_pose):
        x = torch.zeros([input_poses.shape[0], 96]).to(input_poses.device)
        x[:, dim_used] = input_poses
        x[:, index_to_copy] = sample_pose[index_to_copy].unsqueeze(0)
        x[:, index_to_ignore] = x[:, index_to_equal]
        return x

class EncoderTransformer(nn.Module):
    def __init__(self, d_model=3, nhead=1, dim_feedforward=128, dropout=0.1, activation="gelu", num_layers=6, out_dim=48):
        super(EncoderTransformer, self).__init__()
        self.first_linear = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_layer = nn.Linear(d_model*22, out_dim)

    def forward(self, inputs):
        # inputs: (b*T) * 66
        s, d = inputs.shape
        x = inputs.reshape(-1,22,3) # (b*T) * 22 * 3
        x = self.first_linear(x) # (b*T) * 22 * e'
        x = self.transformer_encoder(x).reshape(s, -1) # (b*T) * (22*e')
        x = torch.relu(x)
        return self.embed_layer(x) # (b*T) * e

class DecoderTransformer(nn.Module):
    def __init__(self, d_model=3, nhead=1, dim_feedforward=128, dropout=0.1, activation="gelu", num_layers=6, out_dim=48):
        super(DecoderTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_layer = nn.Linear(d_model*22, out_dim)

    def forward(self, inputs):
        # inputs: (b*T) * e
        x = self.embed_layer(inputs) 
        x = torch.relu(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x).squeeze(1)
        return x

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(66, 66),
            nn.ReLU(),
            nn.Linear(66, 66),
            nn.ReLU()
        )
    def forward(self, x):
        return self.fc(x)+ x

class TransformerAE(nn.Module): 
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args

        self.pre = Preprocess()
        self.post= Postprocess()

        self.nheads=8

        self.encoder = EncoderTransformer(in_dim=96, d_model=96, out_dim=48, num_layers=4, dim_feedforward=128, nhead=self.nheads)
        self.decoder = DecoderTransformer(in_dim=48, d_model=96, out_dim=96, num_layers=4, dim_feedforward=128, nhead=self.nheads)

    def forward(self, inputs):
        x = torch.cat([inputs["observed_pose"].clone(), inputs["future_pose"].clone()], dim=1)
        x = x.reshape(-1, x.shape[-1]) # (b*T) * 96
        sample_pose=x[-1].unsqueeze(0) # 1 * 96

        x = self.pre(x) # (b*T) * 66

        x=self.encoder(x) # (b*T) * e
        out = self.decoder(x) # (b*T) * 66

        out = self.post(out, sample_pose) # (b*T) * 96

        return {
            "pred_pose": inputs["future_pose"].clone(), "out":out
        }


class AE(nn.Module): 
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args

        # self.pre = Preprocess()
        # self.post= Postprocess()
        
        self.encoder = nn.Sequential(
            nn.Linear(96, 66),
            nn.ReLU(),
            nn.Linear(66, 66),
            nn.ReLU(),
            ResBlock(),
            # ResBlock(),
            nn.Linear(66, 32),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 66),
            nn.ReLU(),
            nn.Linear(66, 66),
            nn.ReLU(),
            ResBlock(),
            # ResBlock(),
            nn.Linear(66, 96)
        )


    def forward(self, inputs):
        x = torch.cat([inputs["observed_pose"].clone(), inputs["future_pose"].clone()], dim=1)
        x = x.reshape(-1, x.shape[-1]) # (b*T) * 96
        # sample_pose=x[-1] # 96

        # x = self.pre(x) # (b*T) * 66

        x=self.encoder(x) # (b*T) * e
        out = self.decoder(x) # (b*T) * 66

        # out = self.post(out, sample_pose) # (b*T) * 96

        return {
            "pred_pose": inputs["future_pose"].clone(), "out":out
        }


