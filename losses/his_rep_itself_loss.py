import numpy as np
import torch
import torch.nn as nn
from utils.others import sig5, sigstar

class HisRepItselfLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.output_n = args.output_n
        self.input_n = args.input_n
        self.itera = args.itera
        self.seq_in = args.kernel_size
        self.device = args.device
        self.mode = args.un_mode
        assert args.un_mode in \
            [
                'default', 'ATJ', 'TJ', 'AJ', 'AT', 'A', 'T', 'J', 
                'sig5-T', 'sig5-TJ', 
                'sig5s-T', 'sig5s-TJ', 
                'sigstar-T', 'sigstar-TJ', 
                'sig5r-TJ',
                'sig5shifted-T',
                'input_rel',
                'sig5-TJPrior',
                'sig5-TJPriorSum'
            ]
            
        self.dim = 3
        self.dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                                  26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                  46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                                  75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        # self.connect = [
        #     (11, 12), (12, 13), (13, 14), (14, 15),
        #     (13, 25), (25, 26), (26, 27), (27, 29), (29, 30),
        #     (13, 17), (17, 18), (18, 19), (19, 21), (21, 22),
        #     (1, 2), (2, 3), (3, 4), (4, 5),
        #     (6, 7), (7, 8), (8, 9), (9, 10)
        # ]
        self.connect = [
            (8, 9), (9, 10), (10, 11),
            (9, 17), (17, 18), (18, 19), (19, 20), (20, 21),
            (9, 12), (12, 13), (13, 14), (14, 15), (15, 16),
            (0, 1), (1, 2), (2, 3),
            (4, 5), (5, 6), (6, 7)
        ]
        self.S = np.array([c[0] for c in self.connect])
        self.E = np.array([c[1] for c in self.connect])

        self.sample_rate = 2
        # joints at same loc
        self.joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        self.index_to_ignore = np.concatenate(
            (self.joint_to_ignore * 3, self.joint_to_ignore * 3 + 1, self.joint_to_ignore * 3 + 2))
        self.joint_equal = np.array([13, 19, 22, 13, 27, 30])
        self.index_to_equal = np.concatenate((self.joint_equal * 3, self.joint_equal * 3 + 1, self.joint_equal * 3 + 2))
        
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

    def un_loss(self, pred, gt, params, actions=None, mode='ATJ', pred_disp=1):
        # pred, gt:  B, T, J, D
        # params: A, T, J ---- 16, 25, 22
        B, T, J, D = pred.shape
        # A, T, J = params.shape
        
        if mode == 'input_rel':
            return torch.mean(torch.norm((pred - gt)*params, dim=-1))

        losses = torch.norm(pred - gt, dim=3) # B, T, J
        frames_num = torch.arange(T).to(self.device)
        # v1 : [1, 4, 5, 6, 1, 4, 5, 6, 0, 1, 2, 3, 1, 7, 8, 9, 10, 1, 7, 8, 9, 10]
        # v2: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        joints_num = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).to(self.device)
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
        elif mode == 'sig5-T':
            # params: J, 5
            # torch.arange(T): T,
            s = sig5(params[0, :], frames_num) # 1, T
            s = s.permute(1, 0).unsqueeze(0) # 1, T, 1      

        elif mode == 'sig5-TJ':
            # params: J, 5
            s = sig5(params, frames_num) # J, T
            s = s.permute(1, 0).unsqueeze(0) # 1, T, J

        elif mode == 'sig5s-T':
            s = sig5(params[0, :]**2, frames_num) # 1, T
            s = s.permute(1, 0).unsqueeze(0) # 1, T, 1   

        elif mode == 'sig5s-TJ':
            s = sig5(params**2, frames_num) # J, T
            s = s.permute(1, 0).unsqueeze(0) # 1, T, J

        elif mode == 'sig5shifted-T':
            s = sig5(params + 1.5, frames_num) # J, T
            s = s.permute(1, 0).unsqueeze(0) # 1, T, J

        elif mode == 'sig5r-TJ':
            s = sig5(torch.relu(params) + 0.1, frames_num) # J, T
            s = s.permute(1, 0).unsqueeze(0) # 1, T, J
            
        elif mode == 'sigstar-T':
            params = params[0, :].unsqueeze(0) # 1, 2
            params = torch.cat([params, torch.ones(1, 1).to(self.device)], dim=-1) # 1, 3
            s = sigstar(params, frames_num) # 1, T
            s = s.permute(1, 0).unsqueeze(0)

        elif mode == 'sigstar-TJ':
            # params : J, 2
            params = torch.cat([params, torch.ones(J, 1).to(self.device)], dim=-1) # J, 3
            s = sigstar(params, frames_num) # J, T
            s = s.permute(1, 0).unsqueeze(0)
        elif mode == 'sig5-TJPrior':
            # st = sig5(params[T:], frames_num) # J, T
            # st = st.permute(1, 0).unsqueeze(0)

            # sj = sig5(params[:T], joints_num) # T, J
            # sj = sj.unsqueeze(0)
            st = sig5(params[0], frames_num) # 1, T
            st = st.unsqueeze(-1) # 1, T, 1

            sj = sig5(params[1], joints_num) # 1, J
            sj = sj.unsqueeze(1) # 1, 1, J

            s = st + sj # 1, T, J
        elif mode == 'sig5-TJPriorSum':
            st = sig5(params, frames_num) # J, T
            st = st.permute(1, 0).unsqueeze(0) # 1, T, J

            s = torch.zeros((1, T, J)).to(self.device)
            s[:, :, [0, 4, 8]] = st[:, :, [0, 4, 8]]
            s[:, :, self.E] = st[:, :, self.E]
            for c in self.connect:
                s[:, :, c[1]] = s[:, :, c[0]] + s[:, :, c[1]]
        else:
            raise Exception('The defined uncertainry mode is not supported.')

        
        loss = 1 / torch.exp(s) * losses + s

        if self.args.disp_reg:
            pred_disp = pred_disp.reshape(B, 1, 1)
            loss = loss + self.args.disp_lambda * 1/pred_disp * s

        loss = torch.mean(loss)

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
        pred_disp = None
        if self.args.disp_reg:
            # pred_pose = p3d_out_all[:, :, 0]
            obs_pose = input_data['observed_pose'][:, :, self.dim_used]
            obs_pose = obs_pose.reshape(obs_pose.shape[0], obs_pose.shape[1], len(self.dim_used) // 3, 3)
            pred_disp = torch.norm((obs_pose[:, 1:] - obs_pose[:, :-1]), dim=-1)[:, -self.args.disp_thresh:] # B, T-1, J
            # pred_disp = torch.norm((pred_pose[:, 1:] - pred_pose[:, :-1]), dim=-1) # B, T-1, J
            pred_disp = pred_disp.mean(dim=1).mean(dim=1)
            # pred_disp.requires_grad = False

        if self.mode == 'default':
            if self.itera == 1:
                # print(p3d_out_all.shape)
                loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))
            else:
                # print(7, p3d_out_all.shape)
                loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :self.seq_in+10] - p3d_sup[:, :self.seq_in+10], dim=3))

        elif self.mode == 'input_rel':
            assert self.itera == 1
            params = model_outputs['pred_un_params'][:, :, 0]
            loss_p3d = self.un_loss(pred=p3d_out_all[:, :, 0], gt=p3d_sup, params=params, actions=None, mode=self.mode, pred_disp=pred_disp)
        else:
            if 'A' in self.mode:
                actions = torch.tensor([self.action_dict[a] for a in input_data['action']]).to(self.device)
            else:
                actions = None

            params = model_outputs['un_params']
            if self.itera == 1:
                loss_p3d = self.un_loss(pred=p3d_out_all[:, :, 0], gt=p3d_sup, params=params, actions=actions, mode=self.mode, pred_disp=pred_disp)
            else:
                loss_p3d = self.un_loss(pred=p3d_out_all[:, :self.seq_in+10], gt=p3d_sup[:, :self.seq_in+10], params=params, actions=actions, mode=self.mode, pred_disp=pred_disp)


        p3d_out = model_outputs['pred_metric_pose']
        # print('p3d_out', p3d_out.shape)
        # print('loss mp', p3d_h36[:, -self.output_n:].shape, p3d_out.shape, joints//3)

        mpjpe_p3d_h36 = torch.mean(
            torch.norm(p3d_h36[:, -self.output_n:].reshape(
                [-1, self.output_n, (joints // 3), 3]
            ) - p3d_out.reshape(
                p3d_out.shape[0], p3d_out.shape[1], joints // 3, 3), dim=3
            )
        )
        # print(17, mpjpe_p3d_h36)
        outputs = {'loss': loss_p3d, 'mpjpe': mpjpe_p3d_h36}
        # print(outputs)
        if 'pred_mask' in model_outputs.keys():
            mask_loss = self.bce(model_outputs['pred_mask'], input_data['future_mask'])
            outputs['mask_loss'] = mask_loss

        return outputs