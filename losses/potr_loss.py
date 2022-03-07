import torch
import torch.nn as nn
import torch.nn.functional as F
#from metrics import ADE, FDE

import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")
from models.potr.potr import POTR
from models.potr.data_process import train_preprocess

class POTRLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        if self.args.loss_fn == 'mse':
            self.loss_fn = self.loss_mse
        elif self.args.loss_fn == 'smoothl1':
            self.loss_fn = self.smooth_l1
        elif self.args.loss_fn == 'l1':
            self.loss_fn = self.loss_l1
        else:
            raise ValueError('Unknown loss name {}.'.format(self.args.loss_fn))



    def smooth_l1(self, decoder_pred, decoder_gt):
        l1loss = nn.SmoothL1Loss(reduction='mean')
        return l1loss(decoder_pred, decoder_gt)

    def loss_l1(self, decoder_pred, decoder_gt, reduction='mean'):
        return nn.L1Loss(reduction=reduction)(decoder_pred, decoder_gt)

    def loss_activity(self, logits, class_gt):                                     
        """Computes entropy loss from logits between predictions and class."""
        return nn.functional.cross_entropy(logits, class_gt, reduction='mean')

    def compute_class_loss(self, class_logits, class_gt):
        """Computes the class loss for each of the decoder layers predictions or memory."""
        class_loss = 0.0
        for l in range(len(class_logits)):
            class_loss += self.loss_activity(class_logits[l], class_gt)

        return class_loss/len(class_logits)

    def layerwise_loss_fn(self, decoder_pred, decoder_gt, class_logits=None, class_gt=None):
        """Computes layerwise loss between predictions and ground truth."""
        pose_loss = 0.0

        for l in range(len(decoder_pred)):
            pose_loss += self.loss_fn(decoder_pred[l], decoder_gt)

        pose_loss = pose_loss/len(decoder_pred)
        
        class_loss = None
        if class_logits is not None:
            class_loss = self.compute_class_loss(class_logits, class_gt)
        

        return pose_loss, class_loss

    def ua_loss(self, decoder_pred, decoder_gt, class_logits, class_gt, uncertainty_matrix=None):
        #n_classes = class_logits.shape[-1]
        #n_joints = decoder_gt.shape[-2]

        B = decoder_gt.shape[0]
        T = decoder_gt.shape[-3]
        L = len(decoder_pred)

        pose_loss = 0.0
        class_loss = None
        uncertainty_loss = None

        loss_fn = nn.L1Loss(reduction='none')
        if uncertainty_matrix is not None:
            assert class_gt is not None
            assert uncertainty_matrix.shape == (self.args.num_activities, self.args.n_major_joints)
            uncertainty_vector = uncertainty_matrix[class_gt].reshape(B, 1, self.args.n_major_joints, 1) # (n_joints, )
            u_coeff = (torch.arange(1, T+1) / T).reshape(1, T, 1, 1).to(self.args.device)
        else:
            uncertainty_vector = 1
            u_coeff = 0

        for l in range(L):
            pose_loss += ((1 - u_coeff ** uncertainty_vector) * loss_fn(decoder_pred[l], decoder_gt)).mean()
        
        pose_loss = pose_loss / L

        
        if class_logits is not None:
            class_loss = self.compute_class_loss(class_logits, class_gt)     

        if uncertainty_matrix is not None:
            uncertainty_loss = torch.log(uncertainty_matrix).mean()

        '''for t in range(T)
            for c in range(n_classes):
                for j in range(n_joints):
                    pose_loss += (1 - (t/T)**uncertainty_matrix[c, j]) * self.loss_fn(pred[..., t, c, j], decoder_gt[t, c, j])'''

        return pose_loss, class_loss, uncertainty_loss

    def compute_loss(self, inputs=None, target=None, preds=None, class_logits=None, class_gt=None):
        return self.layerwise_loss_fn(preds, target, class_logits, class_gt)



    def forward(self, model_outputs, input_data):
        input_data = train_preprocess(input_data, self.args)
        
        '''selection_loss = 0
        if self.args.query_selection:
            prob_mat = model_outputs['mat'][-1]
            selection_loss = self.compute_selection_loss(
                inputs=prob_mat, 
                target=input_data['src_tgt_distance']
            )'''

        pred_class, gt_class = None, None
        if self.args.predict_activity:
            gt_class = input_data['action_ids']  # one label for the sequence
            pred_class = model_outputs['out_class']

        uncertainty_matrix = None
        if self.args.consider_uncertainty:
            uncertainty_matrix = model_outputs['uncertainty_matrix']

        """pose_loss, activity_loss = self.compute_loss(
            inputs=input_data['encoder_inputs'],
            target=input_data['decoder_outputs'],
            preds=model_outputs['out_sequences'],
            class_logits=pred_class,
            class_gt=gt_class
        )"""
        '''if not use_uncertainty:
            pose_loss, activity_loss = self.layerwise_loss_fn(
                decoder_pred=model_outputs['out_sequences'], 
                decoder_gt=input_data['decoder_outputs'], 
                class_logits=pred_class, 
                class_gt=gt_class, 
            )
        else:'''
        pose_loss, activity_loss, uncertainty_loss = self.ua_loss(
                decoder_pred=model_outputs['out_sequences'], 
                decoder_gt=input_data['decoder_outputs'], 
                class_logits=pred_class, 
                class_gt=gt_class, 
                uncertainty_matrix=uncertainty_matrix
                )
      
        pl = pose_loss.item()
        step_loss = pose_loss #+ selection_loss

        if self.args.predict_activity:
            step_loss += self.args.activity_weight*activity_loss

        if self.args.consider_uncertainty:
            step_loss += self.args.uncertainty_weight*uncertainty_loss
         
        outputs = {
            'loss': step_loss, 
            #'selection_loss': selection_loss,
            'pose_loss': pl,
            }

        if self.args.predict_activity:
            outputs['activity_loss'] = activity_loss.item()

        if self.args.consider_uncertainty:
            outputs['uncertainty_loss'] = uncertainty_loss.item()

        return outputs


if __name__ == '__main__':
    import os, sys
    thispath = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, thispath+"/../")
    import models.potr.utils as utils

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_activity', default=True)
    parser.add_argument('--activity_weight', default=0.1)
    parser.add_argument('--uncertainty_weight', default=1)
    parser.add_argument('--loss_fn', default='l1')
    parser.add_argument('--query_selection', default=False)
    parser.add_argument('--pad_decoder_inputs', default=True)
    parser.add_argument('--include_last_obs', default=False)
    parser.add_argument('--num_encoder_layers', default=4)
    parser.add_argument('--num_decoder_layers', default=4)
    parser.add_argument('--future_frames_num', default=20)
    parser.add_argument('--obs_frames_num', default=50)
    parser.add_argument('--use_query_embedding', default=False)
    parser.add_argument('--num_layers', default=6)
    parser.add_argument('--model_dim', default=128)
    parser.add_argument('--num_heads', default=2)
    parser.add_argument('--dim_ffn', default=16)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--init_fn_name', default='xavier')
    parser.add_argument('--pre_normalization', default=True)
    parser.add_argument('--pose_dim', default=9)
    parser.add_argument('--pose_embedding_type', default='gcn_enc')
    parser.add_argument('--pos_enc_beta', default=500)
    parser.add_argument('--pos_enc_alpha', default=10)
    parser.add_argument('--use_class_token', default=False)
    parser.add_argument('--num_activities', default=15)
    parser.add_argument('--non_autoregressive', default=True)
    parser.add_argument('--n_major_joints', default=21)
    parser.add_argument('--n_h36m_joints', default=32)
    parser.add_argument('--pose_format', default='rotmat')
    parser.add_argument('--consider_uncertainty', default=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pred_pose_format', default='euler')
    #parser.add_argument('--n_classes', default=15)

    #parser.add_argument('--pad_decoder_inputs', default=True)
    args = parser.parse_args()


    src_seq_length = args.obs_frames_num
    tgt_seq_length = args.future_frames_num
    batch_size = 8

    #pred_seq = [torch.FloatTensor(batch_size, tgt_seq_length, 128).uniform_(0, 1) for l in range(4)]

    inputs = {}
    inputs['observed_rotmat_pose']  = torch.FloatTensor(batch_size, src_seq_length, 32, args.pose_dim).uniform_(0, 1)
    inputs['future_rotmat_pose'] = torch.FloatTensor(batch_size, tgt_seq_length, 32, args.pose_dim).fill_(1)
    inputs['action_ids'] = torch.tensor(list(range(batch_size)))
    #preprocessed_inputs = train_preprocess(inputs, args)
    
    model = POTR(args)
    model_outputs = model(
        inputs,
        None,
        False
        )

    loss_func = POTRLoss(args)

    loss_outputs1 = loss_func(model_outputs, inputs)#, use_uncertainty=False)
    #loss_outputs2 = loss_func(model_outputs, inputs, use_uncertainty=True)
    print(loss_outputs1)
    #print(loss_outputs2)

