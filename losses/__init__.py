from .mse_vel_loss import MSEVelLoss
from .mae_vel_loss import MAEVelLoss

LOSSES = {'mse': MSEVelLoss,
          'mae': MAEVelLoss}
