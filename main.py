from args.dataloader_args import DataloaderArgs
from data_loader.data_loader import get_dataloader

if __name__ == '__main__':
    args = DataloaderArgs('simple_dataset', data_dim=2, is_testing=False, use_mask=False, is_multi_person=False,
                          skip_frame=0,
                          batch_size=None, shuffle=False, pin_memory=False, num_workers=0)
    dataloader = get_dataloader(args)

    for data in dataloader:
        for i, d in enumerate(data):
            print(d.shape)
        break

