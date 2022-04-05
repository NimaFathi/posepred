
import torch
from torch import nn
import numpy as np


def joint_to_index(x):
    return np.concatenate((x * 3, x * 3 + 1, x * 3 + 2))


connect = [
    (11, 12), (12, 13), (13, 14), (14, 15),
    (13, 25), (25, 26), (26, 27), (27, 29), (29, 30),
    (13, 17), (17, 18), (18, 19), (19, 21), (21, 22),
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
        out = out[:, :, E] - out[:, :, S] # B, T, 22, 3
        out = self.xyz_to_spherical(out)
        out = out.reshape(*out.shape[:-2], -1)
        return out

    def xyz_to_spherical(self, inputs):
        # inputs: B, T, 22, 3
        rho = torch.norm(inputs, p=2, dim=-1)
        theta = torch.arctan(inputs[..., 2] / (inputs[..., 0] + 1e-8)).unsqueeze(-1)
        tol = 0
        theta[inputs[..., 0] < tol] = theta[inputs[..., 0] < tol] + torch.pi
        phi = torch.arccos(inputs[..., 1] / (rho + 1e-8)).unsqueeze(-1)
        rho = rho.unsqueeze(-1)
        out = torch.cat([rho, theta, phi], dim=-1)
        out[out.isnan()] = 0
        return out

class Postprocess(nn.Module):
    def __init__(self, args):
        super(Postprocess, self).__init__()
        self.args = args

    def forward(self, observed_pose, pred_pose):
        B, T, D = pred_pose.shape # B, T, 66
        observed_pose = observed_pose.reshape(*observed_pose.shape[:2], 32, 3) # B, T2, 32, 3 
        pred_pose = pred_pose.reshape(B, T, D//3, 3)

        
        x = torch.zeros([B, T, 32, 3]).to(self.args.device)
        x[:, :, index_to_copy] = observed_pose[:, -1:, index_to_copy] # B, T, 32, 3
        
        xyz_vec = self.spherical_to_xyz(pred_pose) # B, T, 22, 3
        x[:, :, E] = xyz_vec
        for c in connect:
            x[:, :, c[1]] = x[:, :, c[0]] + x[:, :, c[1]]
        x[:, :, index_to_ignore] = x[:, :, index_to_equal]
        
        x = x.reshape(B, T, 96)
        return x

    def spherical_to_xyz(self, inputs):
        # inputs: B, T, 22, 3 : rho, theta, phi
        
        x = inputs[..., 0] * torch.sin(inputs[..., 2]) * torch.cos(inputs[..., 1])
        y = inputs[..., 0] * torch.sin(inputs[..., 2]) * torch.sin(inputs[..., 1])
        z = inputs[..., 0] * torch.cos(inputs[..., 2])
        x, y, z = x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)

        return torch.cat([x, z, y], dim=-1)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    import jsonlines
    args = Namespace(device='cuda')
    preproc = Preprocess(args).to(args.device)
    postproc = Postprocess(args).to(args.device)

    path = '/home/zahra/workshop/posepred/preprocessed_data/stanford36m/train_total_xyz.jsonl'
    sequences = []
    with jsonlines.open(path) as reader:
        i = 0
        for seq in reader:
            sequences.append(np.array(seq['xyz_pose']))
            i += 1
            if i == 100: break

    for i in [21]:
        sequence_gt = sequences[i] # T, 96
        inputs = torch.tensor(sequence_gt[:1]).unsqueeze(0).to(args.device).float() # 1, 10, 96
        spherical = preproc(inputs)
        outputs = postproc(inputs, spherical)
        # print(i, torch.allclose(inputs, outputs, atol=1e-4, rtol=1e-4))
        for i in range(32):
            inx = inputs.reshape(32, 3)[i].cpu().detach().numpy()
            outx = outputs.reshape(32, 3)[i].cpu().detach().numpy()
            print(i, np.allclose(inx, outx, atol=1e-4, rtol=1e-4), inx, outx)


        
        #print(i, torch.allclose(inputs, outputs, atol=1e-4, rtol=1e-4))

