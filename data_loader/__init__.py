from torch.utils.data import DataLoader

from .interactive_dataset import InteractiveDataset
from .noisy_solitary_dataset import NoisySolitaryDataset
from .solitary_dataset import SolitaryDataset
from .our_dataset import OurDataset

DATASETS = ['somof_posetrack', 'posetrack', 'somof_3dpw', '3dpw', 'jta', 'jaad', 'pie', 'human3.6m', 'our']
DATA_TYPES = ['train', 'validation', 'test']
VISUALIZING_TYPES = ['observed', 'future', 'predicted', 'completed']


def get_dataloader(dataset_path, args):
    if dataset_path is None:
        return None
    if args.is_interactive:
        dataset = InteractiveDataset(
                dataset_path, 
                args.keypoint_dim, 
                args.persons_num, 
                args.is_testing, 
                args.use_mask,
                args.is_visualizing, 
                args.use_expmap,
                args.use_rotmat,
                args.use_euler,
                args.use_quaternion,
                args.normalize, 
                args.metadata_path
                )
    else:
        if args.loader=="noisy":
            dataset = NoisySolitaryDataset(
                    dataset_path, 
                    args.keypoint_dim, 
                    args.is_testing, 
                    args.use_mask,
                    args.is_visualizing,
                    args.use_expmap,
                    args.use_rotmat,
                    args.use_euler,
                    args.use_quaternion, 
                    args.normalize, 
                    args.metadata_path,
                    args.noise_rate, 
                    args.noise_keypoint, 
                    args.overfit
                    )
        elif args.loader=="ours":
            dataset = OurDataset(
                    dataset_path, 
                    args.keypoint_dim, 
                    args.is_testing, 
                    args.use_mask,
                    args.is_visualizing, 
                    args.use_expmap,
                    args.use_rotmat,
                    args.use_euler,
                    args.use_quaternion, 
                    args.normalize, 
                    args.metadata_path
                    )
        else:
            dataset = SolitaryDataset(
                    dataset_path, 
                    args.keypoint_dim, 
                    args.is_testing, 
                    args.use_mask,
                    args.is_visualizing, 
                    args.use_expmap,
                    args.use_rotmat,
                    args.use_euler,
                    args.use_quaternion, 
                    args.normalize, 
                    args.metadata_path
                    )

    dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=args.shuffle, 
            pin_memory=args.pin_memory,
            num_workers=args.num_workers
            )

    return dataloader