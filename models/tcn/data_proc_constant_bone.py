import numpy as np
import torch
from torch import nn


def joint_to_index(x):
    return np.concatenate((x * 3, x * 3 + 1, x * 3 + 2))


parent = np.array([1, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                   17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

offset = torch.tensor([0.00000000e+00, 1.32948591e+02, 4.42894612e+02, 4.54206447e+02,
                   1.62767078e+02, 7.49994370e+01, 1.32948826e+02, 4.42894413e+02,
                   4.54206590e+02, 1.62767426e+02, 7.49999480e+01, 1.00000000e-01,
                   2.33383263e+02, 2.57077681e+02, 1.21134938e+02, 1.15002227e+02,
                   2.57077681e+02, 1.51034226e+02, 2.78882773e+02, 2.51733451e+02,
                   0.00000000e+00, 9.99996270e+01, 1.00000188e+02, 0.00000000e+00,
                   2.57077681e+02, 1.51031437e+02, 2.78892924e+02, 2.51728680e+02,
                   0.00000000e+00, 9.99998880e+01, 1.37499922e+02, 0.00000000e+00]).view(1, 1, 32, 1).to('cuda')

dim_used = np.array([2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22,
                     25, 26, 27, 29, 30])

index_to_ignore = np.array([16, 20, 23, 24, 28, 31])

index_to_equal = np.array([13, 19, 22, 13, 27, 30])

index_to_copy = np.array([0, 1, 6, 11])


class Preprocess(nn.Module):
    def __init__(self, args):
        super(Preprocess, self).__init__()
        self.args = args

    def forward(self, observed_pose):
        observed_pose = observed_pose.reshape(
            observed_pose.shape[0],
            observed_pose.shape[1],
            32, 3
        )
        observed_pose = observed_pose[:, :, parent, :] - observed_pose
        observed_pose = observed_pose[:, :, dim_used] / offset[:, :, dim_used]
        return observed_pose.reshape(
            observed_pose.shape[0],
            observed_pose.shape[1],
            66
        )


class Postprocess(nn.Module):
    def __init__(self, args):
        super(Postprocess, self).__init__()
        self.args = args

    def forward(self, observed_pose, pred_pose):

        B, T, _ = pred_pose.shape

        pred_pose = pred_pose.reshape(B, T, 22, 3)

        pred_pose = pred_pose / torch.unsqueeze(torch.norm(pred_pose, dim=-1), -1) * offset[:, :, dim_used]

        x = torch.zeros([pred_pose.shape[0], pred_pose.shape[1], 96]).to(self.args.device)
        x[:, :, index_to_copy] = observed_pose[:, -1:, index_to_copy]


        x = x.reshape(B, T, 32, 3)
        x[:, :, dim_used] = pred_pose

        for i in dim_used:
            p = parent[i]
            if p in index_to_ignore:
                p = index_to_equal[index_to_ignore.tolist().index(p)]

            assert (p in dim_used or p in index_to_copy) and p < i
            x[:, :, i] = x[:, :, i] + x[:, :, p]
        x[:, :, index_to_ignore] = x[:, :, index_to_equal]
        return x.reshape(B, T, 96)

# if __name__ == '__main__':
