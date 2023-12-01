from .functions import *
from .dataset_utils import JOINTS_TO_INCLUDE

LOSS_K, UNC_K = 'loss', 'uncertainty'


def entropy_uncertainty(p, as_list=False):
    entropy = - p * torch.log2(p)
    out = torch.nansum(entropy, dim=1)
    if as_list:
        return out
    return torch.mean(out)


def calculate_pose_uncertainty(prediction_pose, dc_model, dataset_name: str) -> dict:
    uncertainties = []
    B, SEQ, NJ = prediction_pose.shape
    for k in range(len(prediction_pose)):
        with torch.no_grad():
            p, _ = dc_model(prediction_pose[k][:, :, JOINTS_TO_INCLUDE[dataset_name]])
            uncertainties.append(entropy_uncertainty(p, as_list=True).cpu().detach().numpy())
    uncertainties = np.concatenate(uncertainties, axis=0)
    return np.mean(uncertainties)
