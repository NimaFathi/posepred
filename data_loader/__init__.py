import os
from torch.utils.data import DataLoader

from path_definition import ROOT_DIR
from .interactive_dataset import InteractiveDataset
from .solitary_dataset import SolitaryDataset
from .noisy_solitary_dataset import NoisySolitaryDataset

DATASETS = ['somof_posetrack', 'posetrack', 'somof_3dpw', '3dpw', 'jta', 'jaad', 'pie', 'human3.6m']
DATA_TYPES = ['train', 'validation', 'test']
VISUALIZING_TYPES = ['observed', 'future', 'predicted', 'completed']


def get_dataloader(dataset_name, args):
    if dataset_name is None:
        return None
    data_folder = os.path.join(ROOT_DIR, 'preprocessed_data')
    dataset_path = os.path.join(data_folder, dataset_name + '.jsonl')
    if args.is_interactive:
        dataset = InteractiveDataset(dataset_path, args.keypoint_dim, args.persons_num, args.is_testing, args.use_mask,
                                     args.is_visualizing, args.use_quaternion, args.normalize, args.metadata_path)
    else:
        if args.is_noisy:
            dataset = NoisySolitaryDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask,
                                           args.is_visualizing, args.use_quaternion, args.noise_rate, args.overfit,
                                           args.noise_keypoint, args.normalize, args.metadata_path)
        else:
            dataset = SolitaryDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask,
                                      args.is_visualizing, args.use_quaternion, args.normalize, args.metadata_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory,
                            num_workers=args.num_workers)
    return dataloader
