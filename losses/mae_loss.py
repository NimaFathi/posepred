import torch.nn as nn


class MAELoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)
