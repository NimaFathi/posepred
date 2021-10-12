import os
from torch.utils.data import DataLoader

from path_definition import ROOT_DIR
from data_loader.interactive_dataset import InteractiveDataset
from data_loader.non_interactive_dataset import NonInteractiveDataset


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
