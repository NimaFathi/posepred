import torch.nn as nn


class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)
