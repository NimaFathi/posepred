from .adam import adam
from .sgd import sgd
from .adamw import adamw
OPTIMIZERS = {'adam': adam,
              'sgd': sgd,
              'adamw': adamw,
              'sam': sam}
