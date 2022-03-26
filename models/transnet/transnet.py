import torch
from torch import nn

class TransNet(nn.Module):
    def __init__(self, args):
        super(TransNet, self).__init__()
        self.args = args
        self.transformer = nn.Transformer(d_model=96, batch_first=True)

    def forward(self, inputs):
        src = inputs['observed_pose']
        tgt = inputs['future_pose']
        out = self.transformer(src, tgt)
        outputs = {
                'pred_pose': out
                }
        return outputs

