from torch.utils.data import DataLoader
from dataloader.basic_dataset import Basic_Dataset
from consts import ROOT_DIR
import os


def basic_dataloader(dataset_name, use_mask, skip_frame, is_multi_person, batch_size, shuffle, pin_memory, num_workers):
    data_folder = os.path.join(ROOT_DIR[:ROOT_DIR.rindex('/')], 'preprocessed_data/')
    dataset_path = data_folder + dataset_name + '.csv'
    dataset = Basic_Dataset(dataset_path, use_mask, skip_frame, is_multi_person)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                            num_workers=num_workers)

    return dataloader


dataloader = basic_dataloader('simple_dataset', True, 0, False, 32, True, False, 1)

for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(dataloader):
    print(idx)
