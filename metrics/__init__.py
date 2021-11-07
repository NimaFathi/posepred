from .pose_metrics import MPJPE, ADE, FDE, VIM, VAM
from .mask_metrics import accuracy, f1_score, precision, recall

POSE_METRICS = {'MPJPE': MPJPE,
                'ADE': ADE,
                'FDE': FDE,
                'VIM': VIM,
                'VAM': VAM}

MASK_METRICS = {'accuracy': accuracy,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall}
