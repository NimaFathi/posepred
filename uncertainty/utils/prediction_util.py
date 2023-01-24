import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import DIM, JOINTS_TO_INCLUDE, INCLUDED_JOINTS_COUNT
from utils.functions import rescale_to_original_joints_count
from model.sts_gcn.sts_gcn import STSGCN

OUT_K, GT_K = 'output', 'gt'
PRED_MODELS = {'sts': STSGCN}
PRED_MODELS_ARGS = {
    'sts': {'input_channels': DIM, 'input_time_frame': 10, 'output_time_frame': 25, 'st_gcnn_dropout': 0.1,
            'joints_to_consider': -1, 'n_txcnn_layers': 4, 'txc_kernel_size': [DIM, DIM], 'txc_dropout': 0.0}}


def get_prediction_model_dict(model, data_loader: DataLoader, input_n: int, output_n: int, dataset_name: str,
                              dev='cuda') -> dict:
    prediction_dict = {OUT_K: [], GT_K: []}
    for _, data_arr in enumerate(data_loader):
        pose = data_arr[0].to(dev)
        B = pose.shape[0]
        inp = pose[:, :input_n, JOINTS_TO_INCLUDE[dataset_name]]. \
            view(B, input_n, INCLUDED_JOINTS_COUNT[dataset_name] // DIM, DIM).permute(0, 3, 1, 2)
        gt = pose[:, input_n:input_n + output_n, :]
        if len(gt) == 1:
            gt = gt.unsqueeze(0)
        with torch.no_grad():
            out = model(inp).permute(0, 1, 3, 2).contiguous().view(-1, output_n, INCLUDED_JOINTS_COUNT[dataset_name])
            out = rescale_to_original_joints_count(out, gt, dataset_name)
            prediction_dict[GT_K].append(gt)
            prediction_dict[OUT_K].append(out)
    prediction_dict[GT_K] = torch.concat(prediction_dict[GT_K], dim=0)
    prediction_dict[OUT_K] = torch.concat(prediction_dict[OUT_K], dim=0)

    return prediction_dict
