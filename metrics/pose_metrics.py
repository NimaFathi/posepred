import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def ADE(pred, target, dim, mask=None):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., d] - target[..., d]) ** 2
    ade = torch.mean(torch.sqrt(displacement))
    return ade


def FDE(pred, target, dim, mask=None):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., -1, :, d] - target[..., -1, :, d]) ** 2
    fde = torch.mean(torch.sqrt(displacement))
    return fde


def local_ade(pred, target, dim, mask=None):
    bs, frames, feat = pred.shape
    keypoints = feat // dim
    pred_pose = pred.reshape(bs, frames, keypoints, dim)
    local_pred_pose = pred_pose - pred_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    target_pose = target.reshape(bs, frames, keypoints, dim)
    local_target_pose = target_pose - target_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    return ADE(local_pred_pose, local_target_pose, dim)


def local_ade_80ms(pred, target, dim, mask=None):
    if pred.shape[1] < 2:
        return 0
    return local_ade(pred[:, :2, :], target[:, :2, :], dim)


def local_ade_160ms(pred, target, dim, mask=None):
    if pred.shape[1] < 4:
        return 0
    return local_ade(pred[:, :4, :], target[:, :4, :], dim)


def local_ade_320ms(pred, target, dim, mask=None):
    if pred.shape[1] < 8:
        return 0
    return local_ade(pred[:, :8, :], target[:, :8, :], dim)

def local_ade_400ms(pred, target, dim, mask=None):
    if pred.shape[1] < 10:
        return 0
    return local_ade(pred[:, :10 , :], target[:, :10, :], dim)

def local_ade_560ms(pred, target, dim, mask=None):
    if pred.shape[1] < 14:
        return 0
    return local_ade(pred[:, :14, :], target[:, :14, :], dim)


def local_ade_720ms(pred, target, dim, mask=None):
    if pred.shape[1] < 18:
        return 0
    return local_ade(pred[:, :18, :], target[:, :18, :], dim)


def local_ade_880ms(pred, target, dim, mask=None):
    if pred.shape[1] < 22:
        return 0
    return local_ade(pred[:, :22, :], target[:, :22, :], dim)


def local_ade_1000ms(pred, target, dim, mask=None):
    if pred.shape[1] < 25:
        return 0
    return local_ade(pred[:, :25, :], target[:, :25, :], dim)

# def local_ade_400(pred, target, dim, mask=None):
#     return local_ade(pred[:, :16, :], target[:, :16, :], dim, mask)
#
# def local_ade_1000(pred, target, dim, mask=None):
#     return local_ade(pred[:, :30, :], target[:, :30, :], dim, mask)
#
# def local_ade_2000(pred, target, dim, mask=None):
#     return local_ade(pred[:, :60, :], target[:, :60, :], dim, mask)

def local_fde(pred, target, dim, mask=None):
    bs, frames, feat = pred.shape
    keypoints = feat // dim
    pred_pose = pred.reshape(bs, frames, keypoints, dim)
    local_pred_pose = pred_pose - pred_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    target_pose = target.reshape(bs, frames, keypoints, dim)
    local_target_pose = target_pose - target_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    return FDE(local_pred_pose, local_target_pose, dim)


def VIM(pred, target, dim, mask):
    """
    Visibilty Ignored Metric
    Inputs:
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        target: Ground truth data - array of shape (pred_len, #joint*(2D/3D))
        dim: dimension of data (2D/3D)
        mask: Visibility mask of pos - array of shape (pred_len, #joint)
    Output:
        errorPose:
    """
    assert mask is not None, 'pred_mask should not be None.'

    target_i_global = np.copy(target)
    if dim == 2:
        mask = np.repeat(mask, 2, axis=-1)
        errorPose = np.power(target_i_global - pred, 2) * mask
        # get sum on joints and remove the effect of missing joints by averaging on visible joints
        errorPose = np.sqrt(np.divide(np.sum(errorPose, 1), np.sum(mask, axis=1)))
        where_are_NaNs = np.isnan(errorPose)
        errorPose[where_are_NaNs] = 0
    elif dim == 3:
        errorPose = np.power(target_i_global - pred, 2)
        errorPose = np.sum(errorPose, 1)
        errorPose = np.sqrt(errorPose)
    else:
        msg = "Dimension of data must be either 2D or 3D."
        logger.error(msg=msg)
        raise Exception(msg)
    return errorPose


def VAM(pred, target, dim, mask, occ_cutoff=100):
    """
    Visibility Aware Metric
    Inputs:
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        target: ground truth data - array of shape (pred_len, #joint*(2D/3D))
        dim: dimension of data (2D/3D)
        mask: Predicted visibilities of pose, array of shape (pred_len, #joint)
        occ_cutoff: Maximum error penalty
    Output:
        seq_err:
    """
    assert mask is not None, 'pred_mask should not be None.'
    assert dim == 2 or dim == 3

    pred_mask = np.repeat(mask, 2, axis=-1)
    seq_err = []
    if type(target) is list:
        target = np.array(target)
    target_mask = np.where(abs(target) < 0.5, 0, 1)
    for frame in range(target.shape[0]):
        f_err = 0
        N = 0
        for j in range(0, target.shape[1], 2):
            if target_mask[frame][j] == 0:
                if pred_mask[frame][j] == 0:
                    dist = 0
                elif pred_mask[frame][j] == 1:
                    dist = occ_cutoff
                    N += 1
            elif target_mask[frame][j] > 0:
                N += 1
                if pred_mask[frame][j] == 0:
                    dist = occ_cutoff
                elif pred_mask[frame][j] == 1:
                    d = np.power(target[frame][j:j + 2] - pred[frame][j:j + 2], 2)
                    d = np.sum(np.sqrt(d))
                    dist = min(occ_cutoff, d)
            else:
                msg = "Target mask must be positive values."
                logger.error(msg)
                raise Exception(msg)
            f_err += dist
        if N > 0:
            seq_err.append(f_err / N)
        else:
            seq_err.append(f_err)
    return np.array(seq_err)
