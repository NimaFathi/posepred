import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def ADE(pred, target, dim):
    """
    Average Displacement Error
    """

    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., d] - target[..., d]) ** 2
    ade = torch.mean(torch.sqrt(displacement))
    return ade


def FDE(pred, target, dim):
    """
    Final Displacement Error
    """
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., -1, :, d] - target[..., -1, :, d]) ** 2
    fde = torch.mean(torch.sqrt(displacement))
    return fde


def local_ade(pred, target, dim):
    bs, frames, feat = pred.shape
    keypoints = feat // dim
    pred_pose = pred.reshape(bs, frames, keypoints, dim)
    local_pred_pose = pred_pose - pred_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    target_pose = target.reshape(bs, frames, keypoints, dim)
    local_target_pose = target_pose - target_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    local_pred_pose = local_pred_pose.reshape(bs, frames, feat)
    local_target_pose = local_target_pose.reshape(bs, frames, feat)
    return ADE(local_pred_pose, local_target_pose, dim)


def local_fde(pred, target, dim):
    bs, frames, feat = pred.shape
    keypoints = feat // dim
    pred_pose = pred.reshape(bs, frames, keypoints, dim)
    local_pred_pose = pred_pose - pred_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    target_pose = target.reshape(bs, frames, keypoints, dim)
    local_target_pose = target_pose - target_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    local_pred_pose = local_pred_pose.reshape(bs, frames, feat)
    local_target_pose = local_target_pose.reshape(bs, frames, feat)
    return FDE(local_pred_pose, local_target_pose, dim)

def MSE(pred, target, dim=None):
    """
    Mean Squared Error
    Arguments:
        pred -- predicted sequence : (batch_size, sequence_length, pose_dim*n_joints)

    """
    # target = target.reshape(*target.shape[:-2], -1)
    assert pred.shape == target.shape
    B, S, D = pred.shape
    mean_errors = torch.zeros((B, S))

    # Training is done in exponential map or rotation matrix or quaternion
    # but the error is reported in Euler angles, as in previous work [3,4,5] 
    for i in np.arange(B):
        # seq_len x complete_pose_dim (H36M==99)
        eulerchannels_pred = pred[i] #.numpy()
        # n_seeds x seq_len x complete_pose_dim (H36M==96)
        action_gt = target#srnn_gts_euler[action]
        
        # seq_len x complete_pose_dim (H36M==96)
        gt_i = action_gt[i]#np.copy(action_gt.squeeze()[i].numpy())
        # Only remove global rotation. Global translation was removed before
        gt_i[:, 0:3] = 0

        # here [2,4,5] remove data based on the std of the batch THIS IS WEIRD!
        # (seq_len, 96) - (seq_len, 96)
        idx_to_use = np.where(np.std(gt_i.detach().cpu().numpy(), 0) > 1e-4)[0]
        euc_error = torch.pow(gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)

        euc_error = torch.sum(euc_error, 1)

        euc_error = torch.sqrt(euc_error)
        mean_errors[i,:] = euc_error

    mean_mean_errors = torch.mean(mean_errors, 0)
    return mean_mean_errors.mean()

