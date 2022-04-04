from .mse_vel import MSEVel
from .mse_pose import MSEPose
from .mae_vel import MAEVel
from .kl_divergence import KLDivergenceLoss
from .mix_and_match_loss import MixAndMatchLoss
from .elbo import ELBO
from .derpof_loss import DeRPoFLoss
from .his_rep_itself_loss import HisRepItselfLoss
from .comp_pred_vel import CompPredVel
from .comp_pred import CompPred
from .trans_cvae import TRANS_CVAE
from .pv_lstm_noisy import PVLSTMNoisy
from .pv_lstm_pro import PVLSTMPro
from .keyplast import Keyplast
from .mpjpe import MPJPE
from .msr_gcn_loss import MSRGCNLoss
from models.sts_gcn.sts_gcn_loss import STSGCNLoss
from models.tcn.tcn_loss import TCNLoss
from .potr_loss import POTRLoss

LOSSES = {'mse_vel': MSEVel,
          'mse_pose': MSEPose,
          'mae_vel': MAEVel,
          'kl_loss': KLDivergenceLoss,
          'm&m': MixAndMatchLoss,
          'elbo': ELBO,
          'derpof': DeRPoFLoss,
          'his_rep_itself': HisRepItselfLoss,
          'comp_pred_vel': CompPredVel,
          'comp_pred': CompPred,
          'trans_cvae': TRANS_CVAE,
          'pv_lstm_noisy': PVLSTMNoisy,
          'pv_lstm_pro': PVLSTMPro,
          'keyplast': Keyplast,
          'mpjpe': MPJPE,
          'msr_gcn':MSRGCNLoss,
          'potr': POTRLoss,
          'sts_gcn': STSGCNLoss,
          'spherical_tcn': TCNLoss     
          }
