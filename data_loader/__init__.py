import os
from torch.utils.data import DataLoader

from path_definition import ROOT_DIR
from .interactive_dataset import InteractiveDataset
from .solitary_dataset import SolitaryDataset
from .noisy_solitary_dataset import NoisySolitaryDataset

DATASETS = ['somof_posetrack', 'posetrack', 'somof_3dpw', '3dpw', 'jta', 'jaad', 'pie', 'human3.6m']
DATA_TYPES = ['train', 'validation', 'test']


def get_dataloader(dataset_name, args):
    if dataset_name is None:
        return None
    data_folder = os.path.join(ROOT_DIR, 'preprocessed_data')
    dataset_path = os.path.join(data_folder, dataset_name + '.jsonl')
    if args.is_interactive:
        dataset = InteractiveDataset(dataset_path, args.keypoint_dim, args.persons_num, args.is_testing, args.use_mask,
                                     args.is_visualizing, args.use_quaternion)
    else:
        if args.is_noisy:
            dataset = NoisySolitaryDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask,
                                           args.is_visualizing, args.use_quaternion, args.noise_rate)
        else:
            dataset = SolitaryDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask,
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
#         self.shuffle = True
#         self.pin_memory = False
#         self.num_workers = 0
#
#
# dl = get_dataloader('JTA_3D_train', Args())
# for i, data in enumerate(dl):
#     pass
