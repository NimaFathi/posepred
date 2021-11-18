from .mask_metrics import accuracy, f1_score, precision, recall
from .pose_metrics import ADE, FDE, local_ade, local_fde, VIM, VAM, local_ade_80ms, local_ade_160ms, \
    local_ade_320ms, local_ade_560ms, local_ade_720ms, local_ade_400ms, local_ade_880ms,  local_ade_1000ms

POSE_METRICS = {'ADE': ADE,
                'FDE': FDE,
                'local_ade': local_ade,
                'local_fde': local_fde,
                'ade_80': local_ade_80ms,
                'ade_160': local_ade_160ms,
                'ade_320': local_ade_320ms,
                'ade_400': local_ade_400ms,
                'ade_560': local_ade_560ms,
                'ade_720': local_ade_720ms,
                'ade_880': local_ade_880ms,
                'ade_1000': local_ade_1000ms,
                'VIM': VIM,
                'VAM': VAM}

MASK_METRICS = {'accuracy': accuracy,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall}
