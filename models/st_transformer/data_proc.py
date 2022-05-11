import numpy as np
import torch
from torch import nn


def joint_to_index(x):
    return np.concatenate((x * 3, x * 3 + 1, x * 3 + 2))


dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                     26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                     46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                     75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

mean = np.array([[[-107.9520, -334.9428, 159.4178, -59.2900, -708.5010, 61.4318,
                   -61.6391, -757.3103, 189.2157, -72.0266, -761.0327, 251.2684,
                   151.7538, -326.8112, 161.4840, 134.0947, -709.5225, 71.7927,
                   153.4157, -744.0466, 200.7195, 163.8421, -737.7441, 261.8018,
                   -17.9565, 210.8857, -12.5731, -30.5735, 429.6271, 36.1767,
                   -43.2606, 489.0777, 114.7583, -54.7775, 578.9327, 88.4990,
                   108.9527, 394.7472, 26.0654, 237.0195, 213.8401, 44.9180,
                   188.2216, 135.0727, 139.9878, 152.3083, 163.3067, 155.1163,
                   196.3242, 118.3158, 182.5405, -163.6815, 375.3079, 23.2578,
                   -266.1268, 186.6490, 53.2938, -217.2098, 156.2352, 160.8916,
                   -200.4095, 191.2718, 165.2301, -223.5151, 149.2325, 211.9896]]])
std = np.array([[[65.4117, 166.9468, 160.5147, 109.2458, 295.7622, 210.9699, 122.5746,
                  308.4443, 228.9709, 131.0754, 310.0372, 235.9644, 74.9162, 174.3366,
                  163.9575, 129.1666, 296.0691, 209.0041, 154.0681, 305.1154, 224.3635,
                  165.3411, 304.1239, 230.2749, 19.6905, 71.2422, 64.0733, 52.6362,
                  150.2302, 141.1058, 68.3720, 177.7844, 164.2342, 78.0215, 203.7356,
                  192.8816, 47.0527, 137.0687, 138.8337, 72.1145, 127.8964, 170.1875,
                  151.9798, 210.0934, 199.3142, 155.3852, 219.3135, 193.1652, 191.3546,
                  254.2903, 225.2465, 45.0912, 135.5994, 133.7429, 74.3784, 133.9870,
                  160.7077, 143.9800, 235.9862, 196.2391, 147.1276, 232.4836, 188.2000,
                  189.1858, 308.0274, 235.1181]]])

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

    def forward(self, observed_pose, pred_pose, normal=True):
        if normal:
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
