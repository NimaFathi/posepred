import torch
from torch import nn

class TransNet(nn.Module):
    def __init__(self, args):
        super(TransNet, self).__init__()
        
        self.transformer = nn.Transformer()
        self.fc = nn.Linear(512, args.pred_frames_num)

    def forward(self, inputs):
        src = inputs['observed_pose']
        print(src.shape)
        tgt = inputs['future_pose']
        print(tgt.shape)
        out = self.transformer(src, tgt)
        print(out.shape)
        out = self.fc(out)
        print(out.shape)
        return out

