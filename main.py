from args.dataloader_args import DataloaderArgs
from data_loader.data_loader import get_dataloader

if __name__ == '__main__':
    dataloader_args = DataloaderArgs('simple_dataset', data_dim=2, use_mask=False, skip_frame=0, batch_size=1, shuffle=False,
                                     pin_memory=False, num_workers=0)

    dataloader = get_dataloader(dataloader_args)
    for data in dataloader:
        print(data[0].shape)

