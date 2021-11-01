from .zero_vel import ZeroVel
from .nearest_neighbor import NearestNeighbor
from .pv_lstm import PVLSTM
from .disentangled import Disentangled
from .derpof import DeRPoF
from .history_repeats_itself import HistoryRepeatsItself
from .mix_and_match import MixAndMatch
from .comp_pred_vel import CompPredVel
from .comp_pred_pose import CompPredPose
from .comp_pred_center import CompPredCenter
from .comp_pred_root import CompPredRoot
from .trans_cvae import TRANS_CVAE
from .pv_lstm_noisy import PVLSTMNoisy

MODELS = {'zero_vel': ZeroVel,
          'nearest_neighbor': NearestNeighbor,
          'pv_lstm': PVLSTM,
          'disentangled': Disentangled,
          'derpof': DeRPoF,
          'history_repeats_itself': HistoryRepeatsItself,
          'mix_and_match': MixAndMatch,
          'comp_pred_vel': CompPredVel,
          'comp_pred_pose': CompPredPose,
          'comp_pred_center': CompPredCenter,
          'comp_pred_root': CompPredRoot,
          'trans_cvae': TRANS_CVAE,
          'pv_lstm_noisy': PVLSTMNoisy,
          }
