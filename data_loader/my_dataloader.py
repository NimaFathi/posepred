import os
from torch.utils.data import DataLoader

from path_definition import ROOT_DIR
from data_loader.interactive_dataset import InteractiveDataset
from data_loader.non_interactive_dataset import NonInteractiveDataset
# from data_loader.hdf5 import NonInteractiveDataset
import psutil

from time import time


def get_dataloader(dataset_name, args):
    if dataset_name is None:
        return None
    data_folder = os.path.join(ROOT_DIR, 'preprocessed_data')
    dataset_path = os.path.join(data_folder, dataset_name + '.jsonl')
    if args.is_interactive:
        dataset = InteractiveDataset(dataset_path, args.keypoint_dim, args.persons_num, args.is_testing, args.use_mask,
                                     args.is_visualizing, args.use_quaternion)
    else:
        dataset = NonInteractiveDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask,
                                        args.is_visualizing, args.use_quaternion)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory,
                            num_workers=args.num_workers)

    return dataloader


class Args:
    def __init__(self):
        self.is_interactive = False
        self.keypoint_dim = 3
        self.is_testing = False
        self.use_mask = True
        self.is_visualizing = False
        self.use_quaternion = False
        self.batch_size = 32
        self.shuffle = False
        self.pin_memory = False
        self.num_workers = 0


# start = time()
# print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
# dl = get_dataloader('JTA_3D_train', Args())
# print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
# print('time:', time() - start)
#
# start = time()
# rows = 0
# for i, data in enumerate(dl):
#     rows += data['observed_mask'].shape[0]
#     # print(i, data.keys())
#     pass
#
# print(rows)
# print('time:', time() - start)
