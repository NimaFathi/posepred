import torch
from torch import nn


class PoseEncoder(nn.Module):
  def __init__(self, n_layers):
    """
    Arguments
      n_layers -- number of convolutional layers
    """
    super(PoseEncoder, self).__init__()

    self.layers = nn.Sequential()

    for i in range(n_layers):
      self.layers.add_module(
          f'conv_{i}',
          nn.Conv2d(
              in_channels=2**i, out_channels=2**(i+1), kernel_size=(3, 1)
              )
          )
      self.layers.add_module(
          f'batchnorm_{i}',
          nn.BatchNorm2d(2**(i+1))
      )
      self.layers.add_module(
          f'relu_{i}',
          nn.ReLU()
      )

  def forward(self, x):
    return self.layers(x)


class PoseDecoder(nn.Module):
  def __init__(self, n_layers):
    """
    Arguments
      n_layers -- number of convolutional layers
    """
    super(PoseDecoder, self).__init__()

    self.layers = nn.Sequential()

    for i in range(n_layers, 0, -1):
      self.layers.add_module(
          f'trans_conv_{i}',
          nn.ConvTranspose2d(
              in_channels=2**i, out_channels=2**(i-1), kernel_size=(3, 1)
              )
          )

      if i > 0:
        self.layers.add_module(
            f'batchnorm_{i}',
            nn.BatchNorm2d(2**(i-1))
        )
        
        self.layers.add_module(
          f'relu_{i}',
          nn.ReLU()
        )

  def forward(self, x):    
    return self.layers(x)


class PoseAutoEncoder(nn.Module):
  def __init__(self, args):
    super(PoseAutoEncoder, self).__init__()
    """
      n_layers -- number of convolutional layers in encoder and decoder
    """
    self.args = args
    self.encoder = PoseEncoder(args.n_layers)
    self.decoder = PoseDecoder(args.n_layers)

  def forward(self, x):
    x = x['observed_pose']
    B, L, D = x.shape
    x = x.unsqueeze(1)
    shape = x.shape
    x = self.encoder(x)
    x = self.decoder(x)
    assert shape == x.shape
    return {'pred_pose': x.squeeze(1)}
