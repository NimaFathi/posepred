import numpy as np
from torch import nn
import torch
from .utils import data_utils

class Proc(nn.Module):
    def __init__(self, args):
        super(Proc, self).__init__()

        self.args = args
        # joints at same loc
        # joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        # index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        # joint_equal = np.array([13, 19, 22, 13, 27, 30])
        # index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
        

        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        # 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)
        self.dim_used = dimensions_to_use
        # self.dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
        #                           26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        #                           46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
        #                           75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        # self.index_to_ignore = index_to_ignore
        # self.index_to_equal = index_to_equal

        self.Index2212 = [[0], [1, 2, 3], [4], [5, 6, 7], [8, 9], [10, 11], [12], [13], [14, 15, 16], [17], [18], [19, 20, 21]]
        self.Index127 = [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11]]
        self.Index74 = [[0, 2], [1, 2], [3, 4], [5, 6]]

    def down(self, x, index):
        N, features, seq_len = x.shape
        my_data = x.reshape(N, -1, 3, seq_len)  # x, 22, 3, 10
        da = torch.zeros((N, len(index), 3, seq_len)) # x, 12, 3, 10
        for i in range(len(index)):
            da[:, i, :, :] = torch.mean(my_data[:, index[i], :, :], dim=1)
        da = da.reshape(N, -1, seq_len)
        return da

    def forward(self, x, preproc):
        if preproc:
            shape = x.shape
            x = x.view((-1, x.shape[-1]))
            x[:, 0:6] = 0
            x = data_utils.expmap2xyz_torch(x)
            x32 = x.view((shape[0], shape[1], -1)).permute((0,2,1))
            x32 = torch.concat([x32, x32[:,:,-1].unsqueeze(-1).repeat(1,1,25)], dim=2)
            print(x32.shape, x32.sum(dim=(0,1)))
            
            x22 = x32[:, self.dim_used, :]
            x12 = self.down(x22, self.Index2212)
            x7 = self.down(x12, self.Index127)
            x4 = self.down(x7, self.Index74)

            # extend inputs + dct + global min and max
            return {
                "p32":x32,
                "p22":x22,
                "p12":x12,
                "p7":x7,
                "p4":x4
            }
        else:
            return x
