from .zero_vel import ZeroVel
from .nearest_neighbor import NearestNeighbor
from .pv_lstm import PVLSTM
from .disentangled import Disentangled
from .derpof import DeRPoF
from  models.history_repeats_itself.history_repeats_itself import HistoryRepeatsItself
from  models.history_repeats_itself.history_repeats_itself_amass import HistoryRepeatsItself as HRI_AMASS
from .mix_and_match import MixAndMatch
from .comp_pred_vel import CompPredVel
from .trans_cvae import TRANS_CVAE
from .pv_lstm_noisy import PVLSTMNoisy
from .pv_lstm_pro import PVLSTMPro
from .keyplast import Keyplast
from .sts_gcn.sts_gcn import STsGCN
from .msr_gcn.msrgcn import MSRGCN
from .potr.potr import POTR
from .st_transformer.CSDI import CSDI_H36M
from .st_transformer.CSDI_PGBIG import CSDI_PGBIG_H36M

MODELS = {'zero_vel': ZeroVel,
          'nearest_neighbor': NearestNeighbor,
          'pv_lstm': PVLSTM,
          'disentangled': Disentangled,
          'derpof': DeRPoF,
          'history_repeats_itself': HistoryRepeatsItself,
          'history_repeats_itself_amass': HRI_AMASS,
          'mix_and_match': MixAndMatch,
          'comp_pred_vel': CompPredVel,
          'trans_cvae': TRANS_CVAE,
          'pv_lstm_noisy': PVLSTMNoisy,
          'pv_lstm_pro': PVLSTMPro,
          'keyplast': Keyplast,
          'potr': POTR,
          'sts_gcn': STsGCN,
          'msr_gcn': MSRGCN,
          'st_transformer': CSDI_H36M,
          'st_transformer_pgbig': CSDI_PGBIG_H36M,
          }
