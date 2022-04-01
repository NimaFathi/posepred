import torch
from torch import nn
from models.spherical_tcn.data_proc_spherical import Preprocess, Postprocess

class CNN_layer(nn.Module): # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout, first_layer = False,
                 bias=True):
        
        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        #padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        #assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        if not first_layer:
            self.block= [
            nn.Conv2d(in_channels + out_channels,out_channels,kernel_size=kernel_size,padding='same') #padding)
            ,nn.PReLU()
            ,nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding='same') #padding)
            ,nn.BatchNorm2d(out_channels)
            # ,nn.Dropout(dropout, inplace=True)
                    ]
        else:
            self.block= [
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding='same') #padding)
            ,nn.BatchNorm2d(out_channels)
            # ,nn.Dropout(dropout, inplace=True)
                    ]

        self.block=nn.Sequential(*self.block)
        

    def forward(self, x):      
        output= self.block(x)
        return output


class SphericalTCN(nn.Module):
    def __init__(self, args):
        super(SphericalTCN,self).__init__()
        
        self.args = args

        input_channels = args.keypoint_dim
        input_time_frame = args.obs_frames_num
        output_time_frame = args.pred_frames_num
        joints_to_consider = args.n_major_joints
        n_txcnn_layers = args.n_txcnn_layers

        st_gcnn_dropout = args.st_gcnn_dropout
        
        txc_kernel_size = args.txc_kernel_size
        txc_dropout = args.txc_dropout

        self.preprocess = Preprocess(args).to(args.device)
        self.postprocess = Postprocess(args).to(args.device)

        for p in self.preprocess.parameters():
            p.requires_grad = False
        
        for p in self.postprocess.parameters():
            p.requires_grad = False


        # txc_kernel_size = [1, 1]

        self.st_gcnns=nn.ModuleList()
        self.n_txcnn_layers=n_txcnn_layers
        self.txcnns=nn.ModuleList()
        self.joint_cnns = nn.ModuleList()
                
                # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)           
        
        self.txcnns.append(CNN_layer(input_time_frame,output_time_frame,txc_kernel_size,txc_dropout, True)) # with kernel_size[3,3] the dimensinons of C,V will be maintained       
        for i in range(1,n_txcnn_layers):
            self.txcnns.append(CNN_layer(input_time_frame,output_time_frame,txc_kernel_size,txc_dropout))
            self.joint_cnns.append(CNN_layer(joints_to_consider,joints_to_consider,txc_kernel_size,txc_dropout, True))

        

    def forward(self, input_dict):
        #B, T, D = input_dict['observed_pose'].shape
        x = self.preprocess(input_dict['observed_pose']) # B, T, 66
        x = x.reshape(x.shape[0], x.shape[1], self.args.n_major_joints, -1)
        #x = input_dict['observed_pose'].reshape(B, T, D//3, 3)
        x = x.permute(0, 1, 3, 2) # B, T, 3, 22

        # x = x.view(-1,
        #            self.args.obs_frames_num,
        #            self.args.n_major_joints,
        #            self.args.keypoint_dim).permute(0, 3, 1, 2) # B, 3, T, 22

        # x = x.permute(0, 2, 1, 3) # B, T, 3, 22

        rho = x[:, :, 0, :].unsqueeze(2) # B, T, 1, 22
        input = x[:, :, 1:, :] # B, T, 2, 22

        y = self.txcnns[0](input)
        #print(y.shape)
        for i in range(1,self.n_txcnn_layers):
            y += self.txcnns[i](torch.cat((input, y), dim=1))
            # y = y.permute(0,3, 2, 1)
            # y += self.joint_cnns[i - 1](y)
            # y = y.permute(0, 3, 2, 1)
        #print(y.shape, rho[:, -1].unsqueeze(1).repeat(1, y.shape[1], 1, 1).shape)
        y = torch.cat([y, rho[:, -1].unsqueeze(1).repeat(1, y.shape[1], 1, 1)], dim=2)

        y = y.permute(0, 1, 3, 2)
        y = y.reshape(-1, self.args.pred_frames_num, self.args.n_major_joints * self.args.keypoint_dim)
        
        outputs = {
            'pred_pose': self.postprocess(input_dict['observed_pose'], y),
            #'pred_metric_pose': self.postprocess(input_dict['observed_metric_pose'], y)
        }

        return outputs