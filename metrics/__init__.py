from .mask_metrics import accuracy, f1_score, precision, recall
from .pose_metrics import ADE, FDE, local_ade, local_fde, VIM, VAM, local_ade_400, local_ade_1000, local_ade_2000

POSE_METRICS = {'ADE': ADE,
                'FDE': FDE,
                'local_ade': local_ade,
                'local_fde': local_fde,
                'ade_400': local_ade_400,
                'ade_1000': local_ade_1000,
                'ade_2000': local_ade_2000,
                'VIM': VIM,
                'VAM': VAM}

MASK_METRICS = {'accuracy': accuracy,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall}
