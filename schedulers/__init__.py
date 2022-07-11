from .reduce_lr_on_plateau import Reduce_LR_On_Plateau
from .step_lr import Step_LR
from .multi_step_lr import MultiStepLR
from .no_step_lr import No_Step_LR

SCHEDULERS = {'reduce_lr_on_plateau': Reduce_LR_On_Plateau,
              'step_lr': Step_LR,
              'multi_step_lr': MultiStepLR,
              'no_step_lr': No_Step_LR
              }
