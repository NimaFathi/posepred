import numpy as np
import torch
import torch.nn as nn
from utils.others import sig5

class HisRepItselfLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.output_n = args.output_n
        self.seq_in = args.kernel_size
        self.device = args.device
        self.mode = args.un_mode
        assert args.un_mode in ['default', 'ATJ', 'TJ', 'AJ', 'AT', 'A', 'T', 'J']
        self.dim = 3
        self.dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                                  26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                  46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                                  75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        self.sample_rate = 2
        # joints at same loc
        self.joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        self.index_to_ignore = np.concatenate(
            (self.joint_to_ignore * 3, self.joint_to_ignore * 3 + 1, self.joint_to_ignore * 3 + 2))
        self.joint_equal = np.array([13, 19, 22, 13, 27, 30])
        self.index_to_equal = np.concatenate((self.joint_equal * 3, self.joint_equal * 3 + 1, self.joint_equal * 3 + 2))
        self.itera = 1
        self.action_dict = {
                "walking": 0, 
                "eating": 1, 
                "smoking": 2, 
                "discussion": 3, 
                "directions": 4,
                "greeting": 5, 
                "phoning": 6, 
                "posing": 7, 
                "purchases": 8, 
                "sitting": 9,
                "sittingdown": 10, 
                "takingphoto": 11, 
                "waiting": 12, 
                "walkingdog": 13,
                "walkingtogether": 14
        }

        # self.idx = np.expand_dims(np.arange(self.seq_in + self.out_n), axis=1) + (
        #         self.out_n - self.seq_in + np.expand_dims(np.arange(self.itera), axis=0))

    def un_loss(self, pred, gt, params, actions=None, mode='ATJ'):
        # pred, gt:  B, T, J, D
        # params: A, T, J ---- 16, 25, 22
        assert mode in ['ATJ', 'TJ', 'AJ', 'AT', 'A', 'T', 'J']
        B, T, J, D = pred.shape
        A, T, J = params.shape

        losses = torch.norm(pred - gt, dim=3) # B, T, J
        if mode == 'ATJ':
            s = params[actions] # B, T, J
        elif mode == 'AT':
            s = params[actions][:, :, 0].unsqueeze(-1) # B, T, 1
        elif mode == 'AJ':
            s = params[actions][:, 0, :].unsqueeze(1) # B, 1, J
        elif mode == 'TJ':
            s = params[0].unsqueeze(0) # 1, T, J #.repeat(B, 1, 1) # B, T, J
        elif mode == 'A':
            s = params[actions][:, 0, 0].reshape(B, 1, 1)
        elif mode == 'T':
            s = params[0, :, 0].reshape(1, T, 1)
        elif mode == 'J':
            s = params[0, 0, :].reshape(1, 1, J)
        elif mode == 'SIG5-T':
            # params: J, 5
            # torch.arange(T): T,
            s = sig5(params[0, :], torch.arange(T)) # 1, T
            s = s.permute(1, 0).unsqueeze(0) # 1, T, 1
        elif mode == 'SIG5-TJ':
            # params: J, 5
            s = sig5(params, torch.arange(T)) # J, T
            s = s.permute(1, 0).unsqueeze(0) # 1, T, J
        
        loss = torch.mean(1 / torch.exp(s) * losses + s)
        return loss


    def forward(self, model_outputs, input_data):
        seq1 = torch.cat((input_data['observed_pose'], input_data['future_pose']), dim=1) # B, T, J*D
        p3d_h36 = seq1.reshape(seq1.shape[0], seq1.shape[1], -1) 
        batch_size, seq_n, joints = p3d_h36.shape
        p3d_h36 = p3d_h36.float().to(self.device)  # todo
        p3d_sup = p3d_h36.clone()[:, :, self.dim_used][:, -self.output_n - self.seq_in:].reshape(
            [-1, self.seq_in + self.output_n, len(self.dim_used) // 3, 3])
        p3d_out_all = model_outputs['pred_pose']
        # print('params', model_outputs['un_params'].shape)
        # print('p3d_out_all', p3d_out_all.shape)
        # print('p3d_sup', p3d_sup.shape)
        # print('p3d_h36', p3d_h36.shape)
        # print('mode', self.mode)
        # print('observed_loss', (p3d_out_all[:, :10, 0] - p3d_sup[:, :10]).sum())
        if self.mode == 'default':
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))
        else:
            if 'SIG5' in self.mode:
                params = model_outputs['sig5_params']
                actions = None
            elif 'A' in self.mode:
                params = model_outputs['un_params']
                actions = torch.tensor([self.action_dict[a] for a in input_data['action']]).to(self.device)
            else:
                params = model_outputs['un_params']
                actions = None

            loss_p3d = self.un_loss(pred=p3d_out_all[:, :, 0], gt=p3d_sup, params=params, actions=actions, mode=self.mode)

        p3d_out = model_outputs['pred_metric_pose']
        # print('p3d_out', p3d_out.shape)
        # print('loss mp', p3d_h36[:, -self.output_n:].shape, p3d_out.shape, joints//3)
        mpjpe_p3d_h36 = torch.mean(
            torch.norm(p3d_h36[:, -self.output_n:].reshape(
                [-1, self.output_n, (joints // 3), 3]
            ) - p3d_out.reshape(
                p3d_out.shape[0], p3d_out.shape[1], p3d_out.shape[2] // 3, 3), dim=3
            )
        )

        outputs = {'loss': loss_p3d, 'mpjpe': mpjpe_p3d_h36}
        # print(outputs)
        if 'pred_mask' in model_outputs.keys():
            mask_loss = self.bce(model_outputs['pred_mask'], input_data['future_mask'])
            outputs['mask_loss'] = mask_loss

        return outputs
