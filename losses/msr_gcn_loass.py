import imp
import torch
import torch.nn as nn
from metrics import ADE
import numpy as np

from models.msr_gcn.utils import data_utils

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return torch.FloatTensor(dct_m), torch.FloatTensor(idct_m)

def reverse_dct_torch(dct_data, idct_m, seq_len):
    '''
    B, 60, 35
    '''
    batch_size, features, dct_n = dct_data.shape

    dct_data = dct_data.permute(2, 0, 1).contiguous().view(dct_n, -1)  # dct_n, B*60
    out_data = torch.matmul(idct_m[:, :dct_n], dct_data).contiguous().view(seq_len, batch_size, -1).permute(1, 2, 0)
    return out_data

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
        da = torch.zeros((N, len(index), 3, seq_len)).to(x.device) # x, 12, 3, 10
        for i in range(len(index)):
            da[:, i, :, :] = torch.mean(my_data[:, index[i], :, :], dim=1)
        da = da.reshape(N, -1, seq_len)
        return da

    def forward(self, x, preproc):
        if preproc:
            shape = x.shape
            x = x.view((-1, x.shape[-1]))
            x[:, 0:6] = 0
            x = data_utils.expmap2xyz_torch(x, x.device)
            x32 = x.view((shape[0], shape[1], -1)).permute((0,2,1))
            # x32 = torch.concat([x32, x32[:,:,-1].unsqueeze(-1).repeat(1,1,25)], dim=2)
            # print(x32.shape, x32.sum(dim=(0,1)))
            
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

def L2NormLoss_train(gt, out):
    '''
    ### (batch size,feature dim, seq len)
    等同于 mpjpe_error_p3d()
    '''

    batch_size, _, seq_len = gt.shape
    gt = gt.view(batch_size, -1, 3, seq_len).permute(0, 3, 1, 2).contiguous()
    out = out.view(batch_size, -1, 3, seq_len).permute(0, 3, 1, 2).contiguous()
    loss = torch.mean(torch.norm(gt - out, 2, dim=-1))
    return loss

def uncertain_loss(gt, out, alphas, lamda, T):
    batch_size, _, seq_len = gt.shape
    gt = gt.view(batch_size, -1, 3, seq_len).permute(0, 1, 3, 2).contiguous()
    out = out.view(batch_size, -1, 3, seq_len).permute(0, 1, 3, 2).contiguous()
    temp = torch.norm(gt-out, 2, dim = -1)
    time_coeff = torch.arange(1,seq_len+1).to(gt.device)/T
    final_coeff = torch.pow(time_coeff, alphas.unsqueeze(-1).repeat(1,1,seq_len))
    # print("alphas", alphas)
    # print("reg", -lamda*torch.log(alphas).mean())
    # print("coeff", final_coeff)
    # print(temp)
    return (temp*(1-final_coeff)).mean()-lamda*torch.log(alphas).mean()

class MSRGCNLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.proc = Proc(args)
        self.args = args
        self.dct_used = args.dct_used
        self.input_n = args.input_n
        self.output_n = args.output_n
        self.dct_m, self.idct_m = get_dct_matrix(self.input_n + self.output_n)
        self.global_min = args.global_min
        self.global_max = args.global_max   
        self.uncertainty_aware = args.uncertainty_aware
        self.lamda = args.lamda
        self.T = args.T

    def forward(self, model_outputs, input_data):
        gt = torch.cat([input_data['observed_expmap_pose'].clone(), input_data['future_expmap_pose'].clone()], dim=1)
    
        gt = gt.reshape((gt.shape[0], gt.shape[1], -1))
        gt = self.proc(gt, True)
        out = model_outputs["pred_pose"]
        losses = {}
        for k in out.keys():
            losses[k] = 0
        frames = [2,4,8,10,14,25]
        
        for k in out.keys():
            temp = out[k]
            temp = (temp+1)/2
            temp = temp *(self.global_max-self.global_min)+self.global_min
            temp = reverse_dct_torch(temp, self.idct_m.to(out[k].device), self.input_n+self.output_n)
            if "22" in k:
                batch_size, _, seq_len = gt[k].shape
                for frame in frames:
                    losses[frame]=torch.mean(torch.norm(gt[k].view(batch_size,-1,3,seq_len)[:,:,:,frame+10-1]- \
                                                        temp.view(batch_size, -1, 3, seq_len)[:,:,:,frame+10-1], 2, -1))
            if self.uncertainty_aware:
                losses[k] += uncertain_loss(gt[k], temp, model_outputs["alphas"][k], self.lamda, self.T)
            else:
                losses[k] += L2NormLoss_train(gt[k], temp)
        
        final_loss = 0
        for k in out.keys():
            final_loss+= losses[k]

        return {'loss': final_loss, 'loss_p22':losses['p22'],'loss_p12':losses['p12'],'loss_p7':losses['p7'],'loss_p4':losses['p4'],
                'loss_1000':losses[25], 'loss_560': losses[14], 'loss_400':losses[10], 'loss_320':losses[8], 'loss_160':losses[4],
                'loss_80':losses[2]}