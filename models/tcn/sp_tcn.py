import torch
from torch import nn
from models.tcn.data_proc import Preprocess, Postprocess
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, T_in, T_out, s_kernel_size, t_kernel_size, c_kernel_size, n_major_joints):
        super(TCN, self).__init__()

        layers = []
        # B, T_in, 22, 3 > B, 3, T_in, 22 > B, 3, T_in, 22     # kernel_size=[t, 1]
        self.conv1 = nn.Conv2d(3, 3, kernel_size=t_kernel_size, padding='same')
        # B, 3, T_in, 22 > B, 22, T_in, 3 > B, 22, T_in, 3
        self.conv2 = nn.Conv2d(n_major_joints, n_major_joints, kernel_size=s_kernel_size, padding='same')
        self.conv3 = nn.Conv2d(n_major_joints, n_major_joints * 2, kernel_size=s_kernel_size, padding='same')
        self.conv4 = nn.Conv2d(n_major_joints * 2, n_major_joints, kernel_size=s_kernel_size, padding='same')
        # B, 22, T_in, 3 > B, T_in, 22, 3 > B, T_out, 22, 3
        self.conv5 = nn.Conv2d(T_in, T_out, kernel_size=c_kernel_size, padding='same')

        # B, T_out, 22, 3 > B, 3, T_out, 22 > B, 3, T_out, 22     # kernel_size=[t, 1]
        self.conv6 = nn.Conv2d(3, 3, kernel_size=t_kernel_size, padding='same')
        # B, 3, T_out, 22 > B, 22, T_out, 3 > B, 22, T_out, 3
        self.conv7 = nn.Conv2d(n_major_joints, n_major_joints, kernel_size=s_kernel_size, padding='same')
        # self.conv8 = nn.Conv2d(n_major_joints, n_major_joints * 2, kernel_size=s_kernel_size, padding='same')
        # self.conv9 = nn.Conv2d(n_major_joints * 2, n_major_joints, kernel_size=s_kernel_size, padding='same')

        



    def forward(self, x):
        # x: B, T_in, 22, 3

        x = x.permute(0, 3, 1, 2) # B, 3, T_in, 22
        x = self.conv1(x) # B, 3, T_in, 22
        x = torch.relu(x)

        x = x.permute(0, 3, 2, 1) # B, 22, T_in, 3
        x = self.conv2(x) # B, 22, T_in, 3
        x = torch.relu(x)

        y = self.conv3(x) # B, 22, T_in, 3
        y = torch.relu(y)

        y = self.conv4(y) # B, 22, T_in, 3
        y = torch.relu(y)

        y = y + x # B, 22, T_in, 3
        y = torch.relu(y)

        y = y.permute(0, 2, 1, 3) # B, T_in, 22, 3
        y = self.conv5(y) # B, T_out, 22, 3

        y = x
        x = x.permute(0, 3, 1, 2) # B, 3, T_in, 22
        x = self.conv6(x) # B, 3, T_in, 22
        x = torch.relu(x)

        x = x.permute(0, 3, 2, 1) # B, 22, T_in, 3
        x = self.conv7(x) # B, 22, T_in, 3

        return x



class SPTCN(nn.Module):
    def __init__(self, args):
        super(SPTCN,self).__init__()
        
        self.args = args

        T_in = args.obs_frames_num
        T_out = args.pred_frames_num
        n_major_joints = args.n_major_joints
        s_kernel_size = args.s_kernel_size
        t_kernel_size = args.t_kernel_size
        c_kernel_size = args.c_kernel_size

        self.preprocess = Preprocess(args).to(args.device)
        self.postprocess = Postprocess(args).to(args.device)

        self.tcn = TCN(T_in, T_out, s_kernel_size, t_kernel_size, c_kernel_size, n_major_joints)

    def forward(self, input_dict):
        x = self.preprocess(input_dict['observed_pose']) # B, T, 66 # observed pose is cartesian coordinate
        x = x.reshape(x.shape[0], x.shape[1], self.args.n_major_joints, self.args.keypoint_dim)

        x = self.tcn(x)

        x = x.reshape(-1, self.args.pred_frames_num, self.args.n_major_joints * self.args.keypoint_dim) # B, T, 66

        outputs = {
            'pred_pose': self.postprocess(input_dict['observed_pose'], x), # B, T, 96
        }
        return outputs