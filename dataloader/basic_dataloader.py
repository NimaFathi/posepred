from torch.utils.data import DataLoader
from dataloader.basic_dataset import Basic_Dataset
from consts import ROOT_DIR
import os


def basic_dataloader(args):
    folder_name = os.path.join(ROOT_DIR[:ROOT_DIR.rindex('/')], 'preprocessed_data/')
    dataset = Basic_Dataset(dataset_path=, use_mask=, skip_frame=)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle, pin_memory=args.pin_memory)
    return dataloader
