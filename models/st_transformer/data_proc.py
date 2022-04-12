import numpy as np
import torch
from torch import nn


def joint_to_index(x):
    return np.concatenate((x * 3, x * 3 + 1, x * 3 + 2))


dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                     26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                     46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                     75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

mean = np.array([[[-108.0207, -357.1349, 162.4628, -60.3845, -749.3369, 54.4319,
                -62.5515, -800.5543, 189.4915, -72.9842, -804.0154, 254.9969,
                150.3320, -345.0031, 167.5580, 133.2313, -745.7170, 69.9487,
                150.4130, -782.4758, 205.3246, 159.4026, -775.5690, 269.4921,
                -17.8942, 222.0518, -13.1888, -30.9244, 453.2491, 38.2384,
                -42.9791, 515.5094, 121.5481, -54.5557, 609.9525, 92.0117,
                108.7500, 415.4085, 26.6593, 236.8900, 221.6402, 45.1059,
                186.8823, 145.6833, 149.6532, 151.0743, 178.7816, 164.5888,
                195.3418, 129.4365, 194.8929, -164.1180, 396.0086, 24.8246,
                -265.7541, 195.1504, 54.7090, -215.6535, 165.0968, 169.1314,
                -198.0947, 202.2790, 171.9553, -220.7997, 158.5216, 222.7706]]])
std = np.array([[[66.3929, 119.1274, 151.6286, 112.3741, 175.5857, 208.0643, 125.0010,
                  168.7289, 222.0870, 132.8225, 168.7699, 224.0900, 75.9225, 136.5678,
                  156.0372, 132.4222, 182.5029, 208.6887, 158.7869, 174.4053, 218.7376,
                  170.3442, 173.4497, 219.1369, 19.7102, 17.9596, 63.2551, 52.4272,
                  51.2776, 137.7591, 68.3521, 84.5895, 155.2936, 78.1808, 85.8814,
                  186.8668, 47.0589, 52.5975, 136.8256, 72.1520, 107.0486, 169.2219,
                  151.7592, 199.0935, 191.9489, 155.2793, 205.1654, 185.0615, 190.2583,
                  245.5722, 214.5396, 44.9131, 56.8072, 131.6514, 73.9480, 119.2110,
                  158.9703, 141.7290, 228.1124, 186.8918, 146.0417, 220.1406, 180.3148,
                  185.6310, 302.1246, 223.1294]]])

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
        self.mean = torch.tensor(mean).to(args.device).float()
        self.std = torch.tensor(std).to(args.device).float()

        print(mean.shape, std.shape)

    def forward(self, observed_pose):
        return (observed_pose[:, :, dim_used] - self.mean) / self.std


class Postprocess(nn.Module):
    def __init__(self, args):
        super(Postprocess, self).__init__()
        self.args = args
        self.mean = torch.tensor(mean).to(args.device).float()
        self.std = torch.tensor(std).to(args.device).float()

        print(mean.shape, std.shape)

    def forward(self, observed_pose, pred_pose):
        pred_pose = (pred_pose * self.std) + self.mean

        x = torch.zeros([pred_pose.shape[0], pred_pose.shape[1], 96]).to(self.args.device)
        x[:, :, dim_used] = pred_pose
        x[:, :, index_to_copy] = observed_pose[:, -1:, index_to_copy]
        x[:, :, index_to_ignore] = x[:, :, index_to_equal]
        return x

# import numpy as np
# import torch
# from torch import nn

#
# def joint_to_index(x):
#     return np.concatenate((x * 3, x * 3 + 1, x * 3 + 2))
#
#
# parent = np.array([1, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
#                    17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1
#
# offset = torch.tensor([0.00000000e+00, 1.32948591e+02, 4.42894612e+02, 4.54206447e+02,
#                    1.62767078e+02, 7.49994370e+01, 1.32948826e+02, 4.42894413e+02,
#                    4.54206590e+02, 1.62767426e+02, 7.49999480e+01, 1.00000000e-01,
#                    2.33383263e+02, 2.57077681e+02, 1.21134938e+02, 1.15002227e+02,
#                    2.57077681e+02, 1.51034226e+02, 2.78882773e+02, 2.51733451e+02,
#                    0.00000000e+00, 9.99996270e+01, 1.00000188e+02, 0.00000000e+00,
#                    2.57077681e+02, 1.51031437e+02, 2.78892924e+02, 2.51728680e+02,
#                    0.00000000e+00, 9.99998880e+01, 1.37499922e+02, 0.00000000e+00]).view(1, 1, 32, 1).to('cuda')
#
# dim_used = np.array([2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22,
#                      25, 26, 27, 29, 30])
#
# index_to_ignore = np.array([16, 20, 23, 24, 28, 31])
#
# index_to_equal = np.array([13, 19, 22, 13, 27, 30])
#
# index_to_copy = np.array([0, 1, 6, 11])
#
#
# class Preprocess(nn.Module):
#     def __init__(self, args):
#         super(Preprocess, self).__init__()
#
#     def forward(self, observed_pose):
#         observed_pose = observed_pose.reshape(
#             observed_pose.shape[0],
#             observed_pose.shape[1],
#             32, 3
#         )
#
#         observed_pose = observed_pose - observed_pose[:, :, parent, :]
#         observed_pose = observed_pose[:, :, dim_used] / offset[:, :, dim_used]
#
#         return observed_pose.reshape(
#             observed_pose.shape[0],
#             observed_pose.shape[1],
#             66
#         )
#
#
# class Postprocess(nn.Module):
#     def __init__(self, args):
#         super(Postprocess, self).__init__()
#
#     def forward(self, observed_pose, pred_pose):
#
#         B, T, _ = pred_pose.shape
#
#         pred_pose = pred_pose.reshape(B, T, 22, 3)
#
#         pred_pose = pred_pose / torch.unsqueeze(torch.norm(pred_pose, dim=-1), -1) * offset[:, :, dim_used]
#
#         x = torch.zeros([B, T, 32, 3]).to('cuda')
#         x[:, :, index_to_copy] = observed_pose.view(B, -1, 32, 3)[:, -1:, index_to_copy]
#         x[:, :, dim_used] = pred_pose
#
#         for i in dim_used:
#             p = parent[i]
#             if p in index_to_ignore:
#                 p = index_to_equal[index_to_ignore.tolist().index(p)]
#
#             assert (p in dim_used or p in index_to_copy) and p < i
#             x[:, :, i] = x[:, :, i] + x[:, :, p]
#         x[:, :, index_to_ignore] = x[:, :, index_to_equal]
#         return x.reshape(B, T, 96)
#
# # if __name__ == '__main__':
