import torch

from data_loader.data_loader import get_dataloader
from args.helper import DataloaderArgs


class NearestNeighbor(torch.nn.Module):
    def __init__(self, args, train_dataset_name):
        super(NearestNeighbor, self).__init__()
        dataloader_args = DataloaderArgs(train_dataset_name, args.keypoint_dim, )
        dataloader = get_dataloader(dataloader_args)

    def forward(self, inputs):
        outputs = []
        shape = inputs[0].shape
        pred_shape = shape[:-2] + (self.args.pred_frames_num, self.output_size)
        pred_vel = torch.zeros(pred_shape)
        outputs.append(pred_vel.to('cuda'))

        if self.args.use_mask:
            mask = inputs[2]
            last_frame = mask[..., -1, :].unsqueeze(-2)
            pred_mask = last_frame.repeat([1 for _ in range(len(mask.shape[:-2]))] + [self.args.pred_frames_num, 1])
            outputs.append(pred_mask.to('cuda'))

        return tuple(outputs)


class DataloaderArgs:
    def __init__(self, dataset_name, keypoint_dim, is_interactive, persons_num, use_mask, skip_num, batch_size,
                 shuffle, pin_memory, num_workers, is_testing=False, is_visualizing=False):
        self.dataset_name = dataset_name
        self.keypoint_dim = keypoint_dim
        self.is_interactive = is_interactive
        self.persons_num = persons_num
        self.use_mask = use_mask
        self.skip_frame = skip_num
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.is_testing = is_testing
        self.is_visualizing = is_visualizing