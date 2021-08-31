import torch
import numpy as np


def accuracy(pred, target):
    zeros = torch.zeros_like(pred)
    ones = torch.ones_like(pred)
    pred = torch.where(pred > 0.5, ones, zeros)
    return torch.sum(pred == target) / torch.numel(pred)


def ADE(pred, target, dim):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., d] - target[..., d]) ** 2
    ade = torch.mean(torch.sqrt(displacement))
    return ade


def FDE(pred, target, dim):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., -1, :, d] - target[..., -1, :, d]) ** 2
    fde = torch.mean(torch.sqrt(displacement))
    return fde


def VIM(pred, target, mask, dim):
    """
    Visibilty Ignored Metric
    Inputs:
        target: Ground truth data - array of shape (pred_len, #joint*(2D/3D))
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        dataset_name: Dataset name
        mask: Visibility mask of pos - array of shape (pred_len, #joint)
    Output:
        errorPose:
    """

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
        raise Exception("Dimension of data must be either 2D or 3D.")
    return errorPose


def VAM(target, pred, occ_cutoff, pred_mask):
    """
    Visibility Aware Metric
    Inputs:
        target: ground truth data - array of shape (pred_len, #joint*(2D/3D))
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        occ_cutoff: Maximum error penalty
        pred_visib: Predicted visibilities of pose, array of shape (pred_len, #joint)
    Output:
        seq_err:
    """
    pred_mask = np.repeat(pred_mask, 2, axis=-1)
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
                raise Exception("Target mask must be positive values.")
            f_err += dist
        if N > 0:
            seq_err.append(f_err / N)
        else:
            seq_err.append(f_err)
    return np.array(seq_err)
