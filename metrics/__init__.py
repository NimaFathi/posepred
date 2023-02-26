from .pose_metrics import ADE, FDE, local_ade, local_fde, VIM, VAM, MSE

POSE_METRICS = {'ADE': ADE,
                'FDE': FDE,
                'local_ade': local_ade,
                'local_fde': local_fde,
                'VIM': VIM,
                'VAM': VAM,
                'MSE': MSE,
                }
