import torch

from model.dc.deep_clustering import DCModel
from utils.functions import *
from utils.dataset_utils import JOINTS_TO_INCLUDE, DIM, SCALE_RATIO, MPJPE_COEFFICIENT
from utils.prediction_util import OUT_K, GT_K

LOSS_K, UNC_K = 'loss', 'uncertainty'


def mpjpe_error(batch_pred, batch_gt):
    batch_pred = batch_pred.contiguous().view(-1, DIM)
    batch_gt = batch_gt.contiguous().view(-1, DIM)
    return torch.mean(torch.norm(batch_gt - batch_pred, 2, 1))


def entropy_uncertainty(p):
    entropy = - p * torch.log2(p)
    out = torch.nansum(entropy, dim=1)
    return torch.mean(out)


def calculate_dict_uncertainty_and_mpjpe(dataset_name: str, model_dict: dict, dc_model: DCModel, batch_size: int,
                                         dev='cuda') -> dict:
    err = 0
    uncertainties = 0
    total_count = len(model_dict[OUT_K])
    num_batches = int(total_count / batch_size)
    if (total_count % batch_size) != 0:
        num_batches += 1
    for k in range(num_batches):
        s_idx, e_idx = k * batch_size, min((k + 1) * batch_size, total_count)
        out = model_dict[OUT_K][s_idx: e_idx, :, :].to(dev)
        gt = model_dict[GT_K][s_idx: e_idx, :, :].to(dev)
        err += mpjpe_error(out, gt).cpu().detach().numpy() * (e_idx - s_idx)
        out = out[:, :, JOINTS_TO_INCLUDE[dataset_name]]
        with torch.no_grad():
            p, _ = dc_model(scale(out, SCALE_RATIO[dataset_name]))
            uncertainties += entropy_uncertainty(p) * (e_idx - s_idx)
    return {LOSS_K: err * MPJPE_COEFFICIENT[dataset_name]/ total_count, UNC_K: uncertainties / total_count}
