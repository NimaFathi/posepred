import torch
import numpy as np


def accuracy(pred, target):
    zeros = torch.zeros_like(pred)
    ones = torch.ones_like(pred)
    pred = torch.where(pred > 0.5, ones, zeros)
    return torch.sum(pred == target) / torch.numel(pred)


def ADE(pred, target, dim):
    b, n, p = pred.size()[0], pred.size()[1], pred.size()[2]
    pred = torch.reshape(pred, (b, n, int(p / dim), dim))
    target = torch.reshape(target, (b, n, int(p / dim), dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[:, :, :, d] - target[:, :, :, d]) ** 2
    ade = torch.mean(torch.mean(torch.sqrt(displacement), dim=1))
    return ade


def FDE(pred, target, dim):
    b, n, p = pred.size()[0], pred.size()[1], pred.size()[2]
    pred = torch.reshape(pred, (b, n, int(p / 2), 2))
    target = torch.reshape(target, (b, n, int(p / 2), 2))
    displacement = torch.sqrt(
        (pred[:, -1, :, 0] - target[:, -1, :, 0]) ** 2 + (
                pred[:, -1, :, 1] - target[:, -1, :, 1]) ** 2)
    fde = torch.mean(torch.mean(displacement, dim=1))
    return fde


def FDE_3d(pred, target):
    b, n, p = pred.size()[0], pred.size()[1], pred.size()[2]
    pred = torch.reshape(pred, (b, n, int(p / 3), 3))
    target = torch.reshape(target, (b, n, int(p / 3), 3))
    displacement = torch.sqrt(
        (pred[:, -1, :, 0] - target[:, -1, :, 0]) ** 2 + (pred[:, -1, :, 1] - target[:, -1, :, 1]) ** 2)
    fde = torch.mean(torch.mean(displacement, dim=1))
    return fde


# TODO: VIM depends on dataset_name!
def VIM(target, pred, dataset_name, mask):
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

    gt_i_global = np.copy(target)
    if dataset_name == "posetrack":
        mask = np.repeat(mask, 2, axis=-1)
        errorPose = np.power(gt_i_global - pred, 2) * mask
        # get sum on joints and remove the effect of missing joints by averaging on visible joints
        errorPose = np.sqrt(np.divide(np.sum(errorPose, 1), np.sum(mask, axis=1)))
        where_are_NaNs = np.isnan(errorPose)
        errorPose[where_are_NaNs] = 0
    else:  # 3dpw
        errorPose = np.power(gt_i_global - pred, 2)
        errorPose = np.sum(errorPose, 1)
        errorPose = np.sqrt(errorPose)
    return errorPose


def VAM(target, pred, occ_cutoff, pred_visib):
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
    pred_visib = np.repeat(pred_visib, 2, axis=-1)
    seq_err = []
    if type(target) is list:
        target = np.array(target)
    GT_mask = np.where(abs(target) < 0.5, 0, 1)
    for frame in range(target.shape[0]):
        f_err = 0
        N = 0
        for j in range(0, target.shape[1], 2):
            if GT_mask[frame][j] == 0:
                if pred_visib[frame][j] == 0:
                    dist = 0
                elif pred_visib[frame][j] == 1:
                    dist = occ_cutoff
                    N += 1
            elif GT_mask[frame][j] > 0:
                N += 1
                if pred_visib[frame][j] == 0:
                    dist = occ_cutoff
                elif pred_visib[frame][j] == 1:
                    d = np.power(target[frame][j:j + 2] - pred[frame][j:j + 2], 2)
                    d = np.sum(np.sqrt(d))
                    dist = min(occ_cutoff, d)
            f_err += dist
        if N > 0:
            seq_err.append(f_err / N)
        else:
            seq_err.append(f_err)
    return np.array(seq_err)
