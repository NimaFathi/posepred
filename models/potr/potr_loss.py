import torch
import torch.nn as nn
#from metrics import ADE, FDE

import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")
from potr.potr import POTR
from potr.preprocess import train_preprocess

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

    def loss_l1(self, decoder_pred, decoder_gt):
        return nn.L1Loss(reduction='mean')(decoder_pred, decoder_gt)

    def loss_activity(self, logits, class_gt):                                     
        """Computes entropy loss from logits between predictions and class."""
        return nn.functional.cross_entropy(logits, class_gt, reduction='mean')


    def layerwise_loss_fn(self, decoder_pred, decoder_gt, class_logits=None, class_gt=None):
        """Computes layerwise loss between predictions and ground truth."""
        pose_loss = 0.0

        for l in range(len(decoder_pred)):
            pose_loss += self.loss_fn(decoder_pred[l], decoder_gt)

        pose_loss = pose_loss/len(decoder_pred)
        if class_logits is not None:
            return pose_loss, self.compute_class_loss(class_logits, class_gt)

        return pose_loss, None

    def compute_loss(self, inputs=None, target=None, preds=None, class_logits=None, class_gt=None):
        return self.layerwise_loss_fn(preds, target, class_logits, class_gt)

    def forward(self, model_outputs, input_data):

        print(input_data['encoder_inputs'].shape, input_data['decoder_outputs'].shape)
        selection_loss = 0
        if self.args.query_selection:
            prob_mat = model_outputs[-1][-1]
            selection_loss = self.compute_selection_loss(
                inputs=prob_mat, 
                target=input_data['src_tgt_distance']
            )

        pred_class, gt_class = None, None
        if self.args.predict_activity:
            gt_class = input_data['action_ids']  # one label for the sequence
            pred_class = model_outputs[1]

        pose_loss, activity_loss = self.compute_loss(
            inputs=input_data['encoder_inputs'],
            target=input_data['decoder_outputs'],
            preds=model_outputs[0],
            class_logits=pred_class,
            class_gt=gt_class
        )

        step_loss = pose_loss + selection_loss
        if self.args.predict_activity:
            step_loss += self.args.activity_weight*activity_loss
            act_loss += activity_loss

         
        outputs = {
            'loss': step_loss, 
            'selection_loss': selection_loss, 
            'activity_loss': activity_loss
            }

        return outputs


if __name__ == '__main__':
    import os, sys
    thispath = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, thispath+"/../")
    import potr.utils as utils

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_activity', default=False)
    parser.add_argument('--activity_weight', default=0)
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
    parser.add_argument('--n_joints', default=21)
    parser.add_argument('--pose_format', default='rotmat')
    #parser.add_argument('--pad_decoder_inputs', default=True)
    args = parser.parse_args()


    src_seq_length = args.obs_frames_num
    tgt_seq_length = args.future_frames_num
    batch_size = 8

    #pred_seq = [torch.FloatTensor(batch_size, tgt_seq_length, 128).uniform_(0, 1) for l in range(4)]

    inputs = {}
    inputs['observed_expmap_pose']  = torch.FloatTensor(batch_size, src_seq_length, 32, args.pose_dim).uniform_(0, 1)
    inputs['future_expmap_pose'] = torch.FloatTensor(batch_size, tgt_seq_length, 32, args.pose_dim).fill_(1)
    
    preprocessed_inputs = train_preprocess(inputs, args)
    print('here16', preprocessed_inputs['encoder_inputs'].shape, preprocessed_inputs['decoder_inputs'].shape)
    model = POTR(args)
    model_outputs = model(preprocessed_inputs['encoder_inputs'],
                       preprocessed_inputs['decoder_inputs'],
                       None,
                       get_attn_weights=False)

    loss_func = POTRLoss(args)

    print('mo', model_outputs[0][0].shape)
    loss_outputs = loss_func(model_outputs, preprocessed_inputs)

    print(loss_outputs)