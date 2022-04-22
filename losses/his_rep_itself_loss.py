import numpy as np
import torch
import torch.nn as nn


class HisRepItselfLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.output_n = args.output_n
        self.input_n = args.input_n
        self.seq_in = args.kernel_size
        self.device = args.device
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
        self.itera = args.itera
        # self.idx = np.expand_dims(np.arange(self.seq_in + self.out_n), axis=1) + (
        #         self.out_n - self.seq_in + np.expand_dims(np.arange(self.itera), axis=0))

    def forward(self, model_outputs, input_data):
        seq1 = torch.cat((input_data['observed_pose'], input_data['future_pose']), dim=1)
        p3d_h36 = seq1.reshape(seq1.shape[0], seq1.shape[1], -1)
        batch_size, seq_n, joints = p3d_h36.shape
        p3d_h36 = p3d_h36.float().to(self.device)  # todo
        p3d_sup = p3d_h36.clone()[:, :, self.dim_used][:, -self.output_n - self.seq_in:].reshape(
            [-1, self.seq_in + self.output_n, len(self.dim_used) // 3, 3])
        p3d_out_all = model_outputs['pred_pose']
        # print(self.itera, p3d_out_all.shape, p3d_sup.shape)
        # print('loss', p3d_out_all[:, :self.seq_in+10].shape, p3d_sup[:, :self.seq_in+10].shape)
        if self.itera == 1:
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))
        else:
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :self.seq_in+10] - p3d_sup[:, :self.seq_in+10], dim=3))

        p3d_out = model_outputs['pred_metric_pose']
        # print(15, p3d_out.shape, p3d_h36.shape)
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
