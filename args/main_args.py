import argparse


class DataloaderArgs:
    def __init__(self, dataset_name, keypoint_dim, is_interactive=False, use_mask=False, is_testing=False, skip_frame=0,
                 batch_size=1, shuffle=True, pin_memory=False, num_workers=0):
        self.dataset_name = dataset_name
        self.keypoint_dim = keypoint_dim
        self.is_interactive = is_interactive
        self.use_mask = use_mask
        self.is_testing = is_testing
        self.skip_frame = skip_frame
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers


def parse_dataloader_args():
    parser = argparse.ArgumentParser('Argument for Dataloader')
    parser.add_argument('--dataset_name', type=str, help='dataset_name')
    parser.add_argument('--keypoint_dim', type=int, help='dimension of each keypoint')
    parser.add_argument('--is_interactive', type=bool, default=False, help='support interaction of people')
    parser.add_argument('--use_mask', type=bool, default=False, help='visibility mask')
    parser.add_argument('--is_testing', type=bool, default=False, help='no ground truth data when testing')
    parser.add_argument('--skip_frame', type=int, default=0, help='skip frame in reading dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    dataloader_args = parser.parse_args()
    return dataloader_args
