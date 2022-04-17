from scipy.interpolate import CubicSpline
import numpy as np

class Interpolate():
    def __init__(self, scale_factor, mode='cubic_spline'):
        assert scale_factor < 1
        self.scale_factor = scale_factor
        self.mode = mode
        if mode == 'cubic_spline':
            self.interpolator = CubicSpline

    def __call__(self, y):
        # y: ..., T, n_major_joints*keypoint_dim
        x = np.arange(y.shape[-2])
        interpolator = self.interpolator(x, y, axis=-2)
        new_x = np.arange(0, y.shape[-2], self.scale_factor)[:y.shape[-2]]
        return interpolator(new_x)


if __name__ == '__main__':
    # x = np.arange(10).reshape(-1, 1)
    x = np.random.randn(5, 10, 20, 2)
    interpolate = Interpolate(0.9)
    print(interpolate(x))


