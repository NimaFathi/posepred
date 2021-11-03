from .reduce_lr_on_plateau import Reduce_LR_On_Plateau
from .step_lr import Step_LR

SCHEDULERS = {'reduce_lr_on_plateau': Reduce_LR_On_Plateau,
              'step_lr': Step_LR,
              }
