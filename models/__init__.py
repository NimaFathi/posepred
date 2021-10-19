from .zero_vel import ZeroVel
from .nearest_neighbor import NearestNeighbor
from .pv_lstm import PVLSTM
from .disentangled import Disentangled
from .derpof import DeRPoF
from .history_repeats_itself import HistoryRepeatsItself
from .mix_and_match import MixAndMatch
from .comp_pred_vel import CompPredVel

MODELS = {'zero_vel': ZeroVel,
          'nearest_neighbor': NearestNeighbor,
          'pv_lstm': PVLSTM,
          'disentangled': Disentangled,
          'derpof': DeRPoF,
          'history_repeats_itself': HistoryRepeatsItself,
          'mix_and_match': MixAndMatch,
          'comp_pred_vel': CompPredVel,
          }
