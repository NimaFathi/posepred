from args.training_args import TrainingArgs
from args.helper import DataloaderArgs
from data_loader.data_loader import get_dataloader

import numpy as np

if __name__ == '__main__':

    args = TrainingArgs(train_dataset_name='train_PoseTrack', valid_dataset_name='train_PoseTrack',
                        model_name='lstm_vel', keypoint_dim=2, epochs=6, load_path=None, is_interactive=True,
                        persons_num=5, use_mask=False, batch_size=3, shuffle=False)

    train_dataloader_args = DataloaderArgs(args.train_dataset_name, args.keypoint_dim, args.is_interactive,
                                           args.persons_num, args.use_mask, args.is_testing, args.skip_frame,
                                           args.batch_size, args.shuffle, args.pin_memory, args.num_workers)

    train_dataloader = get_dataloader(train_dataloader_args)

    for data in train_dataloader:
        pass

    persons = train_dataloader.dataset.persons_stats
    print(args.train_dataset_name)
    print('mean:', np.mean(persons), ' std:', np.std(persons))
