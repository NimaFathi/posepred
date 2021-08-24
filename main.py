from args.training_args import TrainingArgs
from args.helper import DataloaderArgs
from data_loader.data_loader import get_dataloader
import torch

if __name__ == '__main__':
    ones = torch.ones(2, 4, 5, 2)
    zeros = torch.zeros(2, 4, 5, 2)

    displacement = 0
    for d in range(2):
        displacement += (ones[..., d] - zeros[..., d]) ** 2
        print(displacement.shape)
    print(torch.mean(torch.sqrt(displacement)))

    exit()
    args = TrainingArgs(train_dataset_name='sample_interactive', valid_dataset_name='train_PoseTrack',
                        model_name='lstm_vel', keypoint_dim=2, epochs=6, load_path=None, is_interactive=True,
                        persons_num=5, use_mask=False, batch_size=3, shuffle=False)

    train_dataloader_args = DataloaderArgs(args.train_dataset_name, args.keypoint_dim, args.is_interactive,
                                           args.persons_num, args.use_mask, args.is_testing, args.skip_frame,
                                           args.batch_size, args.shuffle, args.pin_memory, args.num_workers)

    train_dataloader = get_dataloader(train_dataloader_args)

    for data in train_dataloader:
        obs_pose, future_vel, = data[0], data[-1]
        print(obs_pose.shape, future_vel.shape)

    # persons = []
    # persons.append(obs_pose.shape[1])
    # print(args.train_dataset_name)
    # print('seq_num:', len(persons), '-->   mean:', np.mean(persons), ' std:', np.std(persons))
