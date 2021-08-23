from args.training_args import TrainingArgs
from args.helper import DataloaderArgs
from data_loader.data_loader import get_dataloader
import numpy as np

if __name__ == '__main__':

    args = TrainingArgs(train_dataset_name='test_JTA',
                        valid_dataset_name='train_PoseTrack',
                        model_name='lstm_vel', keypoint_dim=2, epochs=6, load_path=None, is_interactive=True,
                        batch_size=1, shuffle=False)

    train_dataloader_args = DataloaderArgs(args.train_dataset_name, args.keypoint_dim, args.is_interactive,
                                           args.use_mask, args.is_testing, args.skip_frame, args.batch_size,
                                           args.shuffle, args.pin_memory, args.num_workers)

    train_dataloader = get_dataloader(train_dataloader_args)

    seq = train_dataloader.dataset.seq

    print(seq.shape)

    persons = []

    for data in train_dataloader:
        # obs_pose = data[0]
        # persons.append(obs_pose.shape[1])
        persons.append(data.cpu().numpy())

    print(args.train_dataset_name)
    print('seq_num:', len(persons), '-->   mean:', np.mean(persons), ' std:', np.std(persons))
