from args.dataloader_args import DataloaderArgs
from data_loader.data_loader import get_dataloader

if __name__ == '__main__':
    args = DataloaderArgs('simple_dataset', data_dim=2, is_testing=False, use_mask=False, skip_frame=0,
                          batch_size=1, shuffle=False, pin_memory=False, num_workers=0)
    dataloader = get_dataloader(args)

    for data in dataloader:
        obs_pose = data[0]
        print(obs_pose.shape)
