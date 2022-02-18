import torch
from torch import nn
import numpy as np
import sys
_MAJOR_JOINTS = [
    0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 24, 25, 26, 27
]

def compute_difference_matrix(self, src_seq, tgt_seq):
    """Computes a matrix of euclidean difference between sequences.

    Args:
      src_seq: Numpy array of shape [src_len, dim].
      tgt_seq: Numpy array of shape [tgt_len, dim].

    Returns:
      A matrix of shape [src_len, tgt_len] with euclidean distances.
    """
    B = src_seq.shape[0]
    src_len = src_seq.shape[1] # M
    tgt_len = tgt_seq.shape[1] # N

    distance = np.zeros((B, src_len, tgt_len), dtype=np.float32)
    for b in range(B):
        for i in range(src_len):
            for j in range(tgt_len):
                distance[b, i, j] = np.linalg.norm(src_seq[b, i]-tgt_seq[b, j])

    row_sums = distance.sum(axis=2)
    distance_norm = distance / row_sums[:, :, np.newaxis]
    distance_norm = 1.0 - distance_norm

    return distance, distance_norm

def train_preprocess(inputs, args):
    
    # B, n_frames, n_joints, 9 if rotmat else 3
    obs_pose = inputs['observed_expmap_pose']  
    future_pose = inputs['future_expmap_pose']
    print('here1', obs_pose.shape, future_pose.shape)
    # B, n_frames, 21, 9 or 3
    obs_pose = obs_pose[:, :, _MAJOR_JOINTS]
    future_pose = future_pose[:, :, _MAJOR_JOINTS]
    print('here2', obs_pose.shape, future_pose.shape)

    # B, n_frmas, 21 * joint_dim
    obs_pose = obs_pose.reshape(*obs_pose.shape[:2], -1)
    future_pose = future_pose.reshape(*future_pose.shape[:2], -1)
    print('here3', obs_pose.shape, future_pose.shape)

    src_seq_len = args.obs_frames_num - 1
    if args.include_last_obs:
      src_seq_len += 1

    encoder_inputs = np.zeros((obs_pose.shape[0], src_seq_len, args.pose_dim * args.n_joints), dtype=np.float32)
    decoder_inputs = np.zeros((obs_pose.shape[0], args.future_frames_num, args.pose_dim * args.n_joints), dtype=np.float32)
    decoder_outputs = np.zeros((obs_pose.shape[0], args.future_frames_num, args.pose_dim * args.n_joints), dtype=np.float32)
    print('here5', encoder_inputs.shape, decoder_inputs.shape, decoder_outputs.shape)
    data_sel = torch.cat((obs_pose, future_pose), dim=1)
 
    print('here4', data_sel.shape)

    encoder_inputs[:, :, 0:args.pose_dim * args.n_joints] = data_sel[:, 0:src_seq_len,:]
    decoder_inputs[:, :, 0:args.pose_dim * args.n_joints] = \
        data_sel[:, src_seq_len:src_seq_len + args.future_frames_num, :]

    # source_seq_len = src_seq_len + 1
    decoder_outputs[:, :, 0:args.pose_dim] = data_sel[:, args.obs_frames_num:, 0:args.pose_dim]

    if args.pad_decoder_inputs:
      query = decoder_inputs[:, 0:1, :]
      decoder_inputs = np.repeat(query, args.future_frames_num, axis=1)
      #if self._params['copy_method'] == 'uniform_scan':
      #  copy_uniform_scan(encoder_inputs, decoder_inputs)

    #distance, distance_norm = compute_difference_matrix(
    #    encoder_inputs, decoder_outputs
    #)
    print(encoder_inputs.shape)
    print(decoder_inputs.shape)
    print(decoder_outputs.shape)
    #sys.exit()

    return {
    'encoder_inputs': torch.tensor(encoder_inputs), 
    'decoder_inputs': torch.tensor(decoder_inputs), 
    'decoder_outputs': torch.tensor(decoder_outputs),
    #'actions': action,
    #'action_id': self._action_ids[action],
    #'action_id_instance': [self._action_ids[action]]*target_seq_len,
    #'src_tgt_distance': distance
    }