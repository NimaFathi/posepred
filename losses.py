from models.pv_lstm.mse_vel_loss import MSEVel
from models.nearest_neighbor.mse_pose_loss import MSEPose
from .losses.mae_vel import MAEVel
from .losses.kl_divergence import KLDivergenceLoss
from models.mix_and_matches.mix_and_match_loss import MixAndMatchLoss
from .losses.elbo import ELBO
from models.derpof.derpof_loss import DeRPoFLoss
from models.history_repeats_itself.his_rep_itself_loss import HisRepItselfLoss
from models.comp_pred_vel.comp_pred_vel_loss import CompPredVel
from models.comp_pred_vel.comp_pred_loss import CompPred
from models.trans_cvae.trans_cvae_loss import TRANS_CVAE
from models.pv_lstm.pv_lstm_noisy_loss import PVLSTMNoisy
from models.pv_lstm.pv_lstm_pro_loss import PVLSTMPro
from models.keyplast.keyplast_loss import Keyplast
from models.sts_gcn.mpjpe_loss import MPJPE
from models.msr_gcn.msr_gcn_loass import MSRGCNLoss
from models.potr.potr_loss import POTRLoss

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
          'potr': POTRLoss
          }
