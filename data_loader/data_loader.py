import os
from torch.utils.data import DataLoader

from consts import ROOT_DIR
from data_loader.interactive_dataset import InteractiveDataset
from data_loader.non_interactive_dataset import NonInteractiveDataset


def get_dataloader(args):
    data_folder = os.path.join(ROOT_DIR, 'preprocessed_data/')
    dataset_path = data_folder + args.dataset_name + '.csv'
    if args.is_interactive:
        dataset = InteractiveDataset(dataset_path, args.keypoint_dim, args.num_persons, args.is_testing, args.use_mask,
                                     args.skip_frame)
    else:
        dataset = NonInteractiveDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask,
                                        args.skip_frame)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory,
                            num_workers=args.num_workers)

    return dataloader
