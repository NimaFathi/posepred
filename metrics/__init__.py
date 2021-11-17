from .pose_metrics import ADE, FDE, local_ade, local_fde, VIM, VAM, local_ade_400, local_ade_1000, local_ade_2000
from .mask_metrics import accuracy, f1_score, precision, recall

POSE_METRICS = {'ADE': ADE,
                'FDE': FDE,
                'local_ade': local_ade,
                'ade_400': local_ade_400,
                'ade_1000': local_ade_1000,
                'ade_2000': local_ade_2000,
                'local_fde': local_fde,
                'VIM': VIM,
                'VAM': VAM}

MASK_METRICS = {'accuracy': accuracy,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall}
