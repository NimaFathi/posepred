from .mask_metrics import accuracy, f1_score, precision, recall

MASK_METRICS = {'accuracy': accuracy,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall}
