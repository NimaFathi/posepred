from .mse_vel_loss import MSEVelLoss
from .mae_vel_loss import MAEVelLoss

LOSSES = {'mse_vel_loss': MSEVelLoss,
          'mae_vel_loss': MAEVelLoss}
