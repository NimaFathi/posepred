from torch.utils.data import DataLoader

from .random_crop_dataset import RandomCropDataset

DATASETS = ['3dpw', 'stanford3.6m', 'amass']
DATA_TYPES = ['train', 'validation', 'test']
VISUALIZING_TYPES = ['observed', 'future', 'predicted', 'completed']


def get_dataloader(dataset_path, args):
    if dataset_path is None:
        return None
    
    dataset = RandomCropDataset(
        dataset_path, args.keypoint_dim, args.is_testing, args.use_mask, args.is_visualizing,
        args.model_pose_format, args.metric_pose_format, args.normalize, args.metadata_path,
        args.seq_rate, args.frame_rate, args.len_observed, args.len_future, args.is_h36_testing
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory,
                            num_workers=args.num_workers)
    return dataloader
