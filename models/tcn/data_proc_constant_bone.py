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

bone_length = [
    233.3833, 257.0777, 121.1349, 115.0022, 151.0314, 278.8929, 251.7287,
    99.9999, 170.0182, 151.0342, 278.8828, 251.7335,  99.9996, 141.4212,
    442.8946, 454.2065, 162.7671,  74.9994, 442.8944, 454.2066, 162.7674,
    74.9999
]

parent = np.array([1, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                    17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

offset = np.array(
    [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
        -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
        0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
        0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
        257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
        0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
        0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
        0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])

offset = offset.reshape(-1, 3)
norm_offset = np.linalg.norm(offset, axis=-1)


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
    def __init__(self, args):
        super(Preprocess, self).__init__()
        self.args = args

    def forward(self, observed_pose):
        return observed_pose[:, :, dim_used]


class Postprocess(nn.Module):
    def __init__(self, args):
        super(Postprocess, self).__init__()
        self.args = args

    def forward(self, observed_pose, pred_pose):
        x = torch.zeros([pred_pose.shape[0], pred_pose.shape[1], 96]).to(self.args.device)
        x[:, :, dim_used] = pred_pose
        x[:, :, index_to_copy] = observed_pose[:, -1:, index_to_copy]
        x[:, :, index_to_ignore] = x[:, :, index_to_equal]
        return x

if __name__ == '__main__':
    offset = offset.reshape(-1, 3)
    norm_offset = np.linalg.norm(offset, axis=-1)
    print(norm_offset.shape)
    print(parent.shape)

    connects2 = []
    for i, p in enumerate(parent):
        connects2.append((p, i))

    for i, item in enumerate(connect):
        try:
            idx = connects2.index(item)
            print(norm_offset[idx], bone_length[i])
        except:
            pass

    
    print(connect)
    print(connects2)