from torch.utils.data import DataLoader
from dataloader.basic_dataset import Basic_Dataset
from consts import ROOT_DIR
import os


def basic_dataloader(args):
    data_folder = os.path.join(ROOT_DIR[:ROOT_DIR.rindex('/')], 'preprocessed_data/')
    dataset_path = data_folder + args.dataset_name + '.csv'
    dataset = Basic_Dataset(dataset_path, args.use_mask, args.skip_frame, args.is_multi_person)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory,
                            num_workers=args.num_workers)

    return dataloader




print(dataloader_args.)
dataloader = basic_dataloader(dataloader_args)

for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(dataloader):
    print(idx)
