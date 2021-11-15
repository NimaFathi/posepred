from .pose_metrics import ADE, FDE, local_ade, local_fde, VIM, VAM, new_local_ade
from .mask_metrics import accuracy, f1_score, precision, recall

POSE_METRICS = {'ADE': ADE,
                'FDE': FDE,
                'local_ade': local_ade,
                'local_fde': local_fde,
                'new_local_ade': new_local_ade,
                'VIM': VIM,
                'VAM': VAM}

MASK_METRICS = {'accuracy': accuracy,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall}
