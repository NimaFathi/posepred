from .mask_metrics import accuracy, f1_score, precision, recall
from .pose_metrics import ADE, FDE, local_ade, local_fde, VIM, VAM, local_ade_80ms, local_ade_160ms, local_ade_320ms, \
    local_ade_560ms, local_ade_720ms, local_ade_880ms

POSE_METRICS = {'ADE': ADE,
                'FDE': FDE,
                'local_ade': local_ade,
                'local_fde': local_fde,
                'ade_80': local_ade_80ms,
                'ade_160': local_ade_160ms,
                'ade_320': local_ade_320ms,
                'ade_560': local_ade_560ms,
                'ade_720': local_ade_720ms,
                'ade_880': local_ade_880ms,
                'VIM': VIM,
                'VAM': VAM}

MASK_METRICS = {'accuracy': accuracy,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall}
