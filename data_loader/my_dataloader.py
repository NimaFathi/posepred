import os
from torch.utils.data import DataLoader

from path_definition import ROOT_DIR
from data_loader.interactive_dataset import InteractiveDataset
from data_loader.non_interactive_dataset import NonInteractiveDataset


def get_dataloader(args):
    data_folder = os.path.join(ROOT_DIR, 'preprocessed_data/')
    dataset_path = data_folder + args.dataset_name + '.csv'
    if args.is_interactive:
        dataset = InteractiveDataset(dataset_path, args.keypoint_dim, args.persons_num, args.is_testing, args.use_mask,
                                     args.skip_frame, args.is_visualizing)
    else:
        dataset = NonInteractiveDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask,
                                        args.skip_frame, args.is_visualizing)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory,
                            num_workers=args.num_workers)

    return dataloader


class DataloaderArgs:
    def __init__(self, dataset_name, keypoint_dim, is_interactive, persons_num, use_mask, skip_num, batch_size,
                 shuffle, pin_memory, num_workers, is_testing=False, is_visualizing=False):
        self.dataset_name = dataset_name
        self.keypoint_dim = keypoint_dim
        self.is_interactive = is_interactive
        self.persons_num = persons_num
        self.use_mask = use_mask
        self.skip_frame = skip_num
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.is_testing = is_testing
        self.is_visualizing = is_visualizing


args = DataloaderArgs('3DPW_train', 3, False, 1, False, 0, 20, False, False, 0)

dataloader = get_dataloader(args)

for data in dataloader:
    print(len(data))
    print(data[0].shape)
    exit()
