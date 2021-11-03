from .reduce_lr_on_plateau import reduce_lr_on_plateau
from .step_lr import step_lr

SCHEDULERS = {'reduce_lr_on_plateau': reduce_lr_on_plateau,
              'step_lr': step_lr,
              }
