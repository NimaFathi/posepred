from scipy.interpolate import CubicSpline
import numpy as np
import random
from torch import nn
import torch

class RandomInterpolate(nn.Module):
    def __init__(self, scale_factor, p=0.1, mode='cubic_spline'):
        super(RandomInterpolate, self).__init__()
        assert scale_factor < 1
        self.scale_factor = scale_factor
        self.p = p
        self.mode = mode
        if mode == 'cubic_spline':
            self.interpolator = CubicSpline

    def forward(self, y):
        # y: ..., T, n_major_joints*keypoint_dim
        if random.uniform(0, 1) >= self.p:
            return y
        print('interpolate')
        x = np.arange(y.shape[-2])
        interpolator = self.interpolator(x, y, axis=-2)
        new_x = np.arange(0, y.shape[-2], self.scale_factor)[:y.shape[-2]]
        y = interpolator(new_x)
        return torch.tensor(y)

class RandomFlip():
    pass

if __name__ == '__main__':
    # x = np.arange(10).reshape(-1, 1)
    x = torch.randn(5, 10, 20, 2)
    interpolate = RandomInterpolate(0.9, 0.5)
    print(interpolate(x).shape)


