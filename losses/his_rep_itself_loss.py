import torch
import torch.nn as nn


class HisRepItselfLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.kernel_size = 10
        self.dim = 3

    def forward(self, model_outputs, input_data):
        future_pose = input_data['future_pose']
        feature_n = future_pose.shape[-1] 
        out = model_outputs['pred_pose'] # 179*10*72
        # print("loss:", out.shape, future_pose.shape)
        # print(future_pose.reshape(-1, feature_n // self.dim, self.dim).shape)
        mpjpe = torch.mean(torch.norm(
            future_pose.reshape(-1, feature_n // self.dim, self.dim) - out.reshape(-1, feature_n // self.dim, self.dim),
            dim=2))
        

        observed_pose = input_data['observed_pose']
        sup_seq = torch.cat((observed_pose[:, -self.kernel_size:, :], future_pose), 1).reshape(-1,
                                                                                               feature_n // self.dim,
                                                                                               self.dim)
        
        out_all = model_outputs['out_all'].reshape(-1, feature_n // self.dim, self.dim)
        print(out_all.shape)
        loss_all = torch.mean(torch.norm(out_all - sup_seq, dim=2))

        outputs = {'loss': mpjpe, 'loss_all': loss_all}

        if 'pred_mask' in model_outputs.keys():
            mask_loss = self.bce(model_outputs['pred_mask'], input_data['future_mask'])
            outputs['mask_loss'] = mask_loss

        return outputs
