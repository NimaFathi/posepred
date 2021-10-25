from .mse_vel import MSEVel
from .mae_vel import MAEVel
from .kl_divergence import KLDivergenceLoss
from .mix_and_match_loss import MixAndMatchLoss
from .elbo import ELBO
from .derpof_loss import DeRPoFLoss
from .his_rep_itself_loss import HisRepItselfLoss
from .comp_pred_vel import CompPredVel
from .comp_pred_pose import CompPredPose
from .comp_pred_center import CompPredCenter
from .trans_cvae import TRANS_CVAE
from .pv_lstm_comp import PVLSTMComp

LOSSES = {'mse_vel': MSEVel,
          'mae_vel': MAEVel,
          'kl_loss': KLDivergenceLoss,
          'm&m': MixAndMatchLoss,
          'elbo': ELBO,
          'derpof': DeRPoFLoss,
          'his_rep_itself': HisRepItselfLoss,
          'comp_pred_vel': CompPredVel,
          'comp_pred_pose': CompPredPose,
          'comp_pred_center': CompPredCenter,
          'trans_cvae': TRANS_CVAE,
          'pv_lstm_comp': PVLSTMComp,
          }
