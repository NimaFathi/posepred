import torch
from torch import nn
import numpy as np


def joint_to_index(x):
    return np.concatenate((x * 3, x * 3 + 1, x * 3 + 2))


connect = [
    (11, 12), (12, 13), (13, 14), (14, 15),
    (12, 25), (25, 26), (26, 27), (27, 29), (29, 30),
    (12, 17), (17, 18), (18, 19), (19, 21), (21, 22),
    (1, 2), (2, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9), (9, 10)
]

const = [
    (0, 11),
    (0, 1), 
    (0, 6)
]

S = np.array([c[0] for c in connect])
E = np.array([c[1] for c in connect])

index_to_ignore = np.array([16, 20, 23, 24, 28, 31])
#index_to_ignore = joint_to_index(index_to_ignore)

index_to_equal = np.array([13, 19, 22, 13, 27, 30])
#index_to_equal = joint_to_index(index_to_equal)

index_to_copy = np.array([0, 1, 6, 11])
#index_to_copy = joint_to_index(index_to_copy)

class Preprocess(nn.Module):
    def __init__(self, args):
        super(Preprocess, self).__init__()
        self.args = args

    def forward(self, observed_pose):
        B, T, D = observed_pose.shape
        out = observed_pose.reshape(B, T, D//3, 3) # B, T, 32, 3
        out = out[:, :, E] - out[:, :, S] # B, T, 25, 3
        out = self.xyz_to_spherical(out)
        return out

    def xyz_to_spherical(self, inputs):
        # inputs: B, T, 25, 3
        rho = torch.norm(inputs, dim=-1)
        theta = torch.arctan(inputs[:, :, :, 1] / inputs[:, :, :, 0]).unsqueeze(3)
        phi = torch.arccos(inputs[:, :, :, 2] / rho).unsqueeze(3)
        rho = rho.unsqueeze(3)
        out = torch.cat([rho, theta, phi], dim=3)

        return out


class Postprocess(nn.Module):
    def __init__(self, args):
        super(Postprocess, self).__init__()
        self.args = args

    def forward(self, observed_pose, pred_pose):
        B, T, D = pred_pose.shape # B, T, 66
        observed_pose = observed_pose.reshape(*observed_pose.shape[:2], 32, 3)
        pred_pose = pred_pose.reshape(B, T, D//3, 3)

        
        x = torch.zeros([B, T, 32, 3]).to(self.args.device)
        x[:, :, index_to_copy] = observed_pose[:, -1:, index_to_copy] # B, T, 96
        
        xyz_vec = self.spherical_to_xyz(pred_pose)
        x[:, :, E] = xyz_vec
        for c in connect:
            x[:, :, c[1]] = x[:, :, c[0]] + x[:, :, c[1]]
        x[:, :, index_to_ignore] = x[:, :, index_to_equal]
        
        x = x.reshape(B, T, 96)
        return x

    def spherical_to_xyz(self, inputs):
        # inputs: B, T, 22, 3 : rho, theta, phi
        
        x = inputs[:, :, :, 0] * torch.sin(inputs[:, :, :, 2]) * torch.cos(inputs[:, :, :, 1])
        y = inputs[:, :, :, 0] * torch.sin(inputs[:, :, :, 2]) * torch.sin(inputs[:, :, :, 1])
        z = inputs[:, :, :, 0] * torch.cos(inputs[:, :, :, 2])
        x, y, z = x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)

        return torch.cat([x, y, z], dim=-1)