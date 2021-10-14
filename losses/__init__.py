from .mse_vel import MSEVel
from .mae_vel import MAEVel
from .kl_divergence import KLDivergenceLoss
from .mix_and_match_loss import MixAndMatchLoss
from .elbo import ELBO
from .derpof_loss import DeRPoFLoss
from .his_rep_itself_loss import HisRepItselfLoss

LOSSES = {'mse_vel': MSEVel,
          'mae_vel': MAEVel,
          'kl_loss': KLDivergenceLoss,
          'm&m': MixAndMatchLoss,
          'elbo': ELBO,
          'derpof': DeRPoFLoss,
          'his_rep_itself': HisRepItselfLoss,
          }
