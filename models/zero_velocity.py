import torch


class ZeroVelocity(torch.nn.Module):
    def __init__(self, args):
        super(ZeroVelocity, self).__init__()
        self.args = args
        self.input_size = self.output_size = int(args.keypoints_num * args.keypoint_dim)

    def forward(self, inputs):
        outputs = []
        batch_size = inputs[0].shape[0]
        pred_vel = torch.zeros(batch_size, self.args.pred_frames_num, self.output_size)
        outputs.append(pred_vel.to('cuda'))

        if self.args.use_mask:
            mask = inputs[2]
            last_frame = mask[..., -1, :].unsqueeze(-2)
            pred_mask = last_frame.repeat([1 for _ in range(len(mask.shape[:-2]))] + [self.args.pred_frames_num, 1])
            outputs.append(pred_mask.to('cuda'))

        return tuple(outputs)
