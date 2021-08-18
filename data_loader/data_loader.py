from torch.utils.data import DataLoader
from data_loader.basic_dataset import BasicDataset
from consts import ROOT_DIR
import os


def get_dataloader(args):
    data_folder = os.path.join(ROOT_DIR, 'preprocessed_data/')
    dataset_path = data_folder + args.dataset_name + '.csv'
    dataset = BasicDataset(dataset_path, args.keypoint_dim, args.is_testing, args.use_mask, args.skip_frame)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory,
                            num_workers=args.num_workers)

    return dataloader
