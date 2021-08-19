import torch


class ZeroVel(torch.nn.Module):
    def __init__(self, args):
        super(ZeroVel, self).__init__()
        self.args = args
        self.input_size = self.output_size = int(args.keypoints_num * args.keypoint_dim)

    def forward(self, inputs):
        outputs = []
        pose = inputs[0]

        outputs.append(torch.zeros(pose.shape[0], self.args.pred_frame_num, self.input_size))
        if self.args.use_mask:
            mask = inputs[2]
            outputs.append(mask[:, -1, :].unsqueeze(1).repeat(1, self.args.pred_frame_num, 1))

        return tuple(outputs)
