import torch

from models.lstm_vel import LSTMVel
from models.zero_velocity import ZeroVelocity


class Disentangle1(torch.nn.Module):
    def __init__(self, args):
        super(Disentangle1, self).__init__()
        self.args = args
        self.global_model = ZeroVelocity(args).to(torch.device('cuda'))
        self.local_model = LSTMVel(args).to(torch.device('cuda'))

        self.input_size = self.output_size = int(args.keypoints_num * args.keypoint_dim)



    def forward(self, inputs):
        outputs = []
        batch_size = inputs[0].shape[0]
        outputs.append(torch.zeros(batch_size, self.args.pred_frames_num, self.output_size))
        if self.args.use_mask:
            mask = inputs[2]
            last_frame = mask[..., -1, :].unsqueeze(-2)
            outputs.append(last_frame.repeat([1 for _ in range(len(mask.shape[:-2]))] + [self.args.pred_frames_num, 1]))

        return tuple(outputs)
