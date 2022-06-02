from torch.utils.data import DataLoader

from .interactive_dataset import InteractiveDataset
from .noisy_solitary_dataset import NoisySolitaryDataset
from .random_crop_dataset import RandomCropDataset
from .solitary_dataset import SolitaryDataset

DATASETS = ['somof_posetrack', 'posetrack', 'somof_3dpw',
            '3dpw', 'jta', 'jaad', 'pie', 'human3.6m', 'stanford3.6m', 'amass']
DATA_TYPES = ['train', 'validation', 'test']
VISUALIZING_TYPES = ['observed', 'future', 'predicted', 'completed']


def get_dataloader(dataset_path, args):
    if dataset_path is None:
        return None
    assert args.is_noisy + args.is_solitary + args.is_interactive + args.is_random_crop == 1, \
        "Please specify exactly on dataloader type"
    if args.is_interactive:
        dataset = InteractiveDataset(dataset_path, args.keypoint_dim, args.persons_num, args.is_testing, args.use_mask,
                                     args.is_visualizing, args.use_quaternion, args.normalize, args.metadata_path)
    elif args.is_noisy:
        dataset = NoisySolitaryDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask,
                                       args.is_visualizing, args.use_quaternion, args.normalize, args.metadata_path,
                                       args.noise_rate, args.noise_keypoint, args.overfit)
    elif args.is_random_crop:
        dataset = RandomCropDataset(
            dataset_path, args.keypoint_dim, args.is_testing, args.use_mask, args.is_visualizing,
            args.model_pose_format, args.metric_pose_format, args.normalize, args.metadata_path,
            args.seq_rate, args.frame_rate, args.len_observed, args.len_future, args.is_h36_testing,
            args.displacement_threshold, args.displacement_mode
        )
    elif args.is_solitary:
        dataset = SolitaryDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask,
                                  args.is_visualizing, args.use_quaternion, args.normalize, args.metadata_path)


    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory,
                            num_workers=args.num_workers)
    return dataloader
