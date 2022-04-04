from turtle import forward
import torch
import torch.nn as nn
import sys

class EncoderTransformer(nn.Module):
    def __init__(self, d_model=3, nhead=1, dim_feedforward=128, dropout=0.1, activation="gelu", num_layers=6, out_dim=48, in_dim=96):
        super(EncoderTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_layer = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        x = inputs.reshape(inputs.shape[0], 32, -1)
        x = self.transformer_encoder(x).reshape(inputs.shape[0], -1)
        x = torch.relu(x)
        return self.embed_layer(x)

class DecoderTransformer(nn.Module):
    def __init__(self, d_model=3, nhead=1, dim_feedforward=128, dropout=0.1, activation="gelu", num_layers=6, out_dim=48, in_dim=96):
        super(DecoderTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_layer = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        x = self.embed_layer(inputs)
        x = torch.relu(x)
        x = x.reshape(inputs.shape[0], 32, -1)
        x = self.transformer_encoder(x).reshape(inputs.shape[0], -1)
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

class AE(nn.Module): 
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args
        self.nheads=1
        # self.encoder = EncoderTransformer(in_dim=96, out_dim=48, num_layers=3, dim_feedforward=128, nhead=self.nheads)
        # self.decoder = DecoderTransformer(in_dim=48, out_dim=96, num_layers=3, dim_feedforward=128, nhead=self.nheads)
        self.encoder = nn.Sequential(
            nn.Linear(96, 66),
            nn.ReLU(),
            nn.Linear(66, 66),
            nn.ReLU(),
            ResBlock(),
            nn.Linear(66, 24),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(24, 66),
            nn.ReLU(),
            nn.Linear(66, 66),
            nn.ReLU(),
            ResBlock(),
            nn.Linear(66, 96)
        )


    def forward(self, inputs):
        x = torch.cat([inputs["observed_pose"].clone(), inputs["future_pose"].clone()], dim=1)
        x = x.reshape(-1, x.shape[-1])
        # if self.nheads>1:
        #     x = torch.cat([x for _ in range(self.nheads)], dim=)
        x=self.encoder(x)
        # x = torch.relu(x)
        out = self.decoder(x)

        return {
            "pred_pose": inputs["future_pose"].clone(), "out":out
        }


