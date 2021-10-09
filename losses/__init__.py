from .mse_vel import MSEVel
from .mae_vel import MAEVel
from .kl_divergence import KLDivergenceLoss
from .mix_and_match_loss import MixAndMatchLoss
LOSSES = {'mse_vel': MSEVel,
          'mae_vel': MAEVel,
          'kl_loss': KLDivergenceLoss,
          'mix_and_match_loss': MixAndMatchLoss
          }
