import torch
from torch import nn

from models.tcn.data_proc_constant_bone import Preprocess, Postprocess


class CNN_layer(nn.Module):
    # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 first_layer=False):

        super(CNN_layer, self).__init__()
        self.kernel_size = kernel_size
        padding = (
            (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)  # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        if not first_layer:
            self.block = [
                nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.PReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
            ]
        else:
            self.block = [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            ]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        output = self.block(x)
        return output


class ConstantBoneTcn(nn.Module):
    def __init__(self, args):
        super(ConstantBoneTcn, self).__init__()

        self.args = args

        input_time_frame = args.obs_frames_num
        output_time_frame = args.pred_frames_num
        joints_to_consider = args.n_major_joints
        n_txcnn_layers = args.n_txcnn_layers

        txc_kernel_size = args.txc_kernel_size

        self.preprocess = Preprocess().to(args.device)
        self.postprocess = Postprocess().to(args.device)

        for p in self.preprocess.parameters():
            p.requires_grad = False

        for p in self.postprocess.parameters():
            p.requires_grad = False

        self.st_gcnns = nn.ModuleList()
        self.n_txcnn_layers = n_txcnn_layers
        self.txcnns = nn.ModuleList()
        self.Atxcnns = []
        self.joint_cnns = nn.ModuleList()

        self.txcnns.append(CNN_layer(input_time_frame, output_time_frame, txc_kernel_size, True))
        for i in range(1, n_txcnn_layers):
            self.txcnns.append(CNN_layer(input_time_frame, output_time_frame, txc_kernel_size))
            self.Atxcnns.append(torch.rand(22, 22, requires_grad=True,device=self.args.device))
            self.joint_cnns.append(
                CNN_layer(joints_to_consider, joints_to_consider, txc_kernel_size, True))

    def forward(self, input_dict):
        B, _, _ = input_dict['observed_pose'].shape
        x = self.preprocess(input_dict['observed_pose'])  # B, T, 66 # observed pose is cartesian coordinate
        x = x.reshape(x.shape[0], x.shape[1], self.args.n_major_joints, -1)
        x = x.permute(0, 1, 3, 2)  # B, T, 3, 22

        y = self.txcnns[0](x)
        for i in range(1, self.n_txcnn_layers):
            y = y + self.txcnns[i](torch.einsum('IJ,BTCJ->BTCI', self.Atxcnns[i - 1], torch.cat((x, y), dim=1)))
            y = y.permute(0, 3, 2, 1)
            y = y + self.joint_cnns[i - 1](y)
            y = y.permute(0, 3, 2, 1)

        y = y.permute(0, 1, 3, 2)  # B, T, 22, 3
        y = y.reshape(-1, self.args.pred_frames_num, self.args.n_major_joints * self.args.keypoint_dim)  # B, T, 66

        outputs = {
            'pred_pose': self.postprocess(input_dict['observed_pose'], y),  # B, T, 96
        }
        # outputs = {
        #     'pred_pose' : y
        # }
        return outputs
