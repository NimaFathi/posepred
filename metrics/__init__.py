from .mask_metrics import accuracy, f1_score, precision, recall
from .pose_metrics import ADE, FDE, VIM, VAM

MASK_METRICS = {'accuracy': accuracy,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall}

POSE_METRICS = {'ADE': ADE,
                'FDE': FDE,
                'VIM': VIM,
                'VAM': VAM}
