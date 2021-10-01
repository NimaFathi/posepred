from .mse_vel import MSEVel
from .mae_vel import MAEVel

LOSSES = {'mse_vel': MSEVel,
          'mae_vel': MAEVel}
