import os
from torch.utils.data import DataLoader

from path_definition import ROOT_DIR
from data_loader.interactive_dataset import InteractiveDataset
from data_loader.non_interactive_dataset import NonInteractiveDataset
# from data_loader.hdf5 import NonInteractiveDataset

from time import time


def get_dataloader(dataset_name, args):
    if dataset_name is None:
        return None
    data_folder = os.path.join(ROOT_DIR, 'preprocessed_data')
    dataset_path = os.path.join(data_folder, dataset_name + '.h5')
    if args.is_interactive:
        dataset = InteractiveDataset(dataset_path, args.keypoint_dim, args.persons_num, args.is_testing, args.use_mask,
                                     args.is_visualizing, args.use_quaternion)
    else:
        dataset = NonInteractiveDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask,
                                        args.is_visualizing, args.use_quaternion)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory,
                            num_workers=args.num_workers)

    return dataloader


# class Args:
#     def __init__(self):
#         self.is_interactive = False
#         self.keypoint_dim = 3
#         self.is_testing = False
#         self.use_mask = True
#         self.is_visualizing = False
#         self.use_quaternion = False
#         self.batch_size = 32
#         self.shuffle = False
#         self.pin_memory = False
#         self.num_workers = 0
#
#
# start = time()
# dl = get_dataloader('train_16_14_1_JTA', Args())
# print('time:', time() - start)
#
# start = time()
# for i, data in enumerate(dl):
#     # print(i, data.keys())
#     pass
# print('time:', time() - start)
