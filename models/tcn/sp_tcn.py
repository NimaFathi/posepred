from pandas import lreshape
import torch
from torch import nn
from models.tcn.data_proc import Preprocess, Postprocess
import torch.nn.functional as F

class TemporalLayer(nn.Module):
    def __init__(self, n_channels, ksize, use_activation=True):
        super(TemporalLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=ksize, groups=n_channels)
        self.activation = nn.ReLU()
        self.use_activation = use_activation
        
    def forward(self, x):
        # x: B, T, J, D
        x = x.permute(0, 2, 3, 1) # B, J, D, T
        x = self.conv(x) # B, J, D, T
        if self.use_activation:
            x = self.activation(x)
        x = x.permute(0, 3, 1, 2) # B, T, J, D

        return x
        
class SpacialLayer(nn.Module):
    def __init__(self, n_channels, ksize=[1, 1], use_activation=True):
        super(SpacialLayer, self).__init__()
        layers = [
            nn.Conv2d(in_channels=n_channels, out_channels=int(1.5*n_channels), kernel_size=ksize),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(1.5*n_channels), out_channels=n_channels, kernel_size=ksize),
        ]
        if use_activation:
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: B, T, J, D
        x = x.permute(0, 2, 3, 1) # B, J, D, T
        x = self.layers(x) # B, J, D, T
        x = x.permute(0, 3, 1, 2) # B, T, J, D

        return x

class D3Layer(nn.Module):
    def __init__(self, n_channels=3, ksize=[1, 1], use_activation=True):
        super(D3Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=ksize)
        self.activation = nn.ReLU()
        self.use_activation = use_activation

    def forward(self, x):
        # x: B, T, J, D
        x = x.permute(0, 3, 2, 1) # B, D, J, T
        x = self.conv(x) # B, D, J, T
        if self.use_activation:
            x = self.activation(x)
        x = x.permute(0, 3, 2, 1) # B, T, J, D

        return x       

class SPLayer(nn.Module):
    def __init__(self, n_joints, tksize, sksize, use_activation=True):
        super(SPLayer, self).__init__()
        self.layers = nn.Sequential(
            TemporalLayer(n_joints, tksize),
            SpacialLayer(n_joints, sksize),
            D3Layer(use_activation=use_activation)
        )
        
    def forward(self, x):
        return self.layers(x)


class SPTCN(nn.Module):
    def __init__(self, args):
        super(SPTCN, self).__init__()
        
        self.args = args

        self.preprocess = Preprocess(args).to(args.device)
        self.postprocess = Postprocess(args).to(args.device)

        #self.tcn = TCN(args.T_in, args.T_out, args.s_kernel_size, args.t_kernel_size, args.c_kernel_size,args. n_major_joints)
        layers = []

        for i in range(args.n_layers - 1):
            layers.append(SPLayer(args.n_major_joints, args.tksize, args.sksize))
        
        layers.append(SPLayer(args.n_major_joints, args.tksize, args.sksize, use_activation=False))

        self.layers = nn.Sequential(*layers)
        print(self.layers)

    def forward(self, input_dict):
        x = self.preprocess(input_dict['observed_pose']) # B, T, 66 # observed pose is cartesian coordinate
        x = x.reshape(x.shape[0], x.shape[1], self.args.n_major_joints, self.args.keypoint_dim)
        
        x = self.layers(x)
        
        x = x.reshape(-1, self.args.pred_frames_num, self.args.n_major_joints * self.args.keypoint_dim) # B, T, 66

        outputs = {
            'pred_pose': self.postprocess(input_dict['observed_pose'], x), # B, T, 96
        }
        return outputs