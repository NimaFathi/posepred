from models.zero_vel import ZeroVel
from models.nearest_neighbor import NearestNeighbor
from models.pv_lstm import PVLSTM
from models.disentangled import Disentangled
from models.derpof import DeRPoF
from models.his_rep_itself import HisRepItself

models = {'zero_vel': ZeroVel,
          'nearest_neighbor': NearestNeighbor,
          'pv_lstm': PVLSTM,
          'disentangled': Disentangled,
          'derpof': DeRPoF,
          'his_rep_itself': HisRepItself}
