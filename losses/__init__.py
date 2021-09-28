from .mse_loss import MSELoss
from .mae_loss import MAELoss
from .bce_loss import BCELoss

LOSSES = {'mse': MSELoss,
          'mae': MAELoss,
          'bce': BCELoss}
