import torch
from tqdm import tqdm
import pandas as pd

from .functions import *
from .dataset_utils import JOINTS_TO_INCLUDE, DIM, SCALE_RATIO, MPJPE_COEFFICIENT, ALL_JOINTS_COUNT, TRAIN_K, TEST_K
from .prediction_util import OUT_K, GT_K, INP_K, RJ, NRJ, ABS, DIFF, RT, CMP, ABS_ROC, DIFF_ROC, CLS_MAP, IDX

LOSS_K, UNC_K = 'loss', 'uncertainty'


def get_uncertainty(z, gt):
    z_encoded = z.cpu().detach().numpy()
    x = np.array([(z_encoded - g) for g in gt.cpu().detach().numpy()])
    norm_x = np.linalg.norm(x, axis=2)
    uncertainties = np.min(norm_x, axis=0)
    return uncertainties


def mpjpe_error(batch_pred, batch_gt, as_list=False):
    b, l, k = batch_gt.shape
    batch_pred = batch_pred.contiguous().view(-1, DIM)
    batch_gt = batch_gt.contiguous().view(-1, DIM)
    n = torch.norm(batch_gt - batch_pred, 2, 1)
    if as_list:
        n = n.view(b, l * k // DIM)
        return torch.mean(n, axis=1)
    return torch.mean(n)


def entropy_uncertainty(p, as_list=False):
    entropy = - p * torch.log2(p)
    out = torch.nansum(entropy, dim=1)
    if as_list:
        return out
    return torch.mean(out)


def calculate_dict_uncertainty_and_mpjpe(dataset_name: str, model_dict: dict, dc_model, batch_size: int,
                                         dev='cuda') -> dict:
    err = []
    uncertainties = []
    #    uncertainties = 0
    total_count = len(model_dict[OUT_K])
    num_batches = int(total_count / batch_size)
    if (total_count % batch_size) != 0:
        num_batches += 1
    for k in range(num_batches):
        s_idx, e_idx = k * batch_size, min((k + 1) * batch_size, total_count)
        out = model_dict[OUT_K][s_idx: e_idx, :, :].to(dev)
        gt = model_dict[GT_K][s_idx: e_idx, :, :].to(dev)
        err.append(mpjpe_error(out, gt, as_list=True).cpu().detach().numpy())
        out = out[:, :, JOINTS_TO_INCLUDE[dataset_name]]
        with torch.no_grad():
            out = scale(out, SCALE_RATIO[dataset_name])
            p, _ = dc_model(out)
            #            d = maha(out)
            #            uncertainties.append(d.cpu().detach().numpy())
            uncertainties.append(entropy_uncertainty(p, as_list=True).cpu().detach().numpy())
    uncertainties = np.concatenate(uncertainties, axis=0)
    err = np.concatenate(err, axis=0)
    err = err * MPJPE_COEFFICIENT[dataset_name]
    #    torch.save({'err': err, 'uncertainties': uncertainties}, 'walking/e4.pt')
    return {LOSS_K: np.mean(err), UNC_K: np.mean(uncertainties)}


def calculate_dataloader_dict_uncertainty(dataset_name: str, loader_dict: dict, dc_model, batch_size: int, std,
                                          ds: str, dev='cuda', ret_avg=True):
    uncertainties = []
    total_count = len(loader_dict[GT_K])
    num_batches = int(total_count / batch_size)
    if (total_count % batch_size) != 0:
        num_batches += 1
    for k in range(num_batches):
        s_idx, e_idx = k * batch_size, min((k + 1) * batch_size, total_count)
        gt = loader_dict[GT_K][s_idx: e_idx, :, :].to(dev)
        if gt.shape[-1] == ALL_JOINTS_COUNT[dataset_name]:
            gt = gt[:, :, JOINTS_TO_INCLUDE[dataset_name]]
        with torch.no_grad():
            scale_ratio = SCALE_RATIO[dataset_name] if ds == TEST_K else 1
            gt = scale(gt, scale_ratio)
            #            gt = scale(gt, scale_ratio)
            #            d = maha(gt)
            #            uncertainties.append(d.cpu().detach().numpy())
            p, _ = dc_model(gt)
            idx = dc_model.predict(gt).cpu().detach().numpy().astype(int)
            uncertainties.append(entropy_uncertainty(p, as_list=True).cpu().detach().numpy())
    #            print(type(idx), idx, std[idx])
    #            uncertainties.append(entropy_uncertainty(p, as_list=True).cpu().detach().numpy() * std[idx])
    uncertainties = np.concatenate(uncertainties, axis=0)
    #    pct = np.percentile(uncertainties, 90)
    #    uncertainties = uncertainties[np.where(uncertainties <= pct)]
    if ret_avg:
        return {UNC_K: np.mean(uncertainties)}
    else:
        return uncertainties
