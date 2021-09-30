from .adam import adam
from .sgd import sgd

MODELS = {'adam': adam,
          'sgd': sgd}
