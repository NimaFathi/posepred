import argparse


class Dataloader_Args:
    def __init__(self, dataset_name, use_mask=False, batch_size=1, shuffle=True, pin_memory=False, num_workers=1):
        self.dataset_name = dataset_name
        self.use_mask = use_mask
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers


def dataloader_parse_args():
    parser = argparse.ArgumentParser('Argument for Dataloader.')
    parser.add_argument('--dataset_name', type=str, help='dataset_name')
    parser.add_argument('--use_mask', type=bool, default=False, help='visibility mask')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    dataloader_args = parser.parse_args()
    return dataloader_args
