import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from .dataset_utils import DIM, JOINTS_TO_INCLUDE, INCLUDED_JOINTS_COUNT
from .functions import rescale_to_original_joints_count
from ..model.sts_gcn.sts_gcn import STSGCN

OUT_K, GT_K = 'outputs', 'ground_truths'
PRED_MODELS = {'sts': STSGCN}
PRED_MODELS_ARGS = {
    'sts': {'input_channels': DIM, 'input_time_frame': 10, 'output_time_frame': 25, 'st_gcnn_dropout': 0.1,
            'joints_to_consider': -1, 'n_txcnn_layers': 4, 'txc_kernel_size': [DIM, DIM], 'txc_dropout': 0.0}}


def get_prediction_model_dict(model, data_loader: DataLoader, input_n: int, output_n: int, dataset_name: str,
                              dev='cuda') -> dict:
    prediction_dict = {OUT_K: [], GT_K: []}
    pose_key = None
    for data_arr in data_loader:
        gt = data_arr["future_pose"]
        B = gt.shape[0]
        if len(gt) == 1:
            gt = gt.unsqueeze(0)
        with torch.no_grad():
            out = model(dict(data_arr))["pred_pose"]
            prediction_dict[GT_K].append(gt)
            prediction_dict[OUT_K].append(out)
    prediction_dict[GT_K] = torch.concat(prediction_dict[GT_K], dim=0)
    prediction_dict[OUT_K] = torch.concat(prediction_dict[OUT_K], dim=0)

    return prediction_dict
