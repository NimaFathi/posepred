from .zero_vel import ZeroVel
from .nearest_neighbor import NearestNeighbor
from .pv_lstm import PVLSTM
from .disentangled import Disentangled
from .derpof import DeRPoF
from .history_repeats_itself import HistoryRepeatsItself

MODELS = {'zero_vel': ZeroVel,
          'nearest_neighbor': NearestNeighbor,
          'pv_lstm': PVLSTM,
          'disentangled': Disentangled,
          'derpof': DeRPoF,
          'history_repeats_itself': HistoryRepeatsItself}


def get_model(model_args):
    model_name = model_args.model_name
    assert model_name in MODELS.keys(), "invalid model"
    return MODELS[model_name](model_args)
