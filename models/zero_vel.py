import torch


class ZeroVel(torch.nn.Module):
    def __init__(self, args):
        super(ZeroVel, self).__init__()
        self.args = args
        self.input_size = self.output_size = int(args.keypoints_num * args.keypoint_dim)

    def forward(self, inputs):
        outputs = []
        shape = inputs[0].shape
        pred_shape = shape[:-2] + tuple([self.args.pred_frames_num, self.output_size])
        pred_vel = torch.zeros(pred_shape)
        outputs.append(pred_vel.to('cuda'))

        if self.args.use_mask:
            mask = inputs[2]
            last_frame = mask[..., -1, :].unsqueeze(-2)
            pred_mask = last_frame.repeat([1 for _ in range(len(mask.shape[:-2]))] + [self.args.pred_frames_num, 1])
            outputs.append(pred_mask.to('cuda'))

        return tuple(outputs)
