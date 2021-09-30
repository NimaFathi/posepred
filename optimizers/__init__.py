from .adam import adam
from .sgd import sgd

OPTIMIZERS = {'adam': adam,
              'sgd': sgd}
