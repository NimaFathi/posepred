import torch.nn as nn


def L1():
    return nn.L1Loss()


def MSE():
    return nn.MSELoss()


def BCE():
    return nn.BCELoss()
