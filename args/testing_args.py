import argparse

from args.helper import DataloaderArgs, ModelArgs


def parse_testing_args():
    args = __parse_testing_args()
    dataloader_args = DataloaderArgs(args.dataset_name, args.keypoint_dim, args.interactive, args.persons_num,
                                     args.use_mask, args.skip_frame, args.batch_size, args.shuffle, args.pin_memory,
                                     args.num_workers, is_testing=True)
    model_args = ModelArgs(args.model_name, args.use_mask, args.keypoint_dim)

    return dataloader_args, model_args, args.load_path, args.pred_frames_num, args.interactive


def __parse_testing_args():
    parser = argparse.ArgumentParser('Testing Arguments')

    # dataloader_args
    parser.add_argument('--dataset_name', type=str, help='test_dataset_name')
    parser.add_argument('--keypoint_dim', type=int, help='dimension of each keypoint')
    parser.add_argument('--pred_frames_num', type=int, help='number of future frames to predict')
    parser.add_argument('--interactive', default=False, action='store_true', help='consider interaction')
    parser.add_argument('--persons_num', type=int, default=1, help='number of people in each sequence')
    parser.add_argument('--use_mask', default=False, action='store_true', help='use visibility mask')
    parser.add_argument('--skip_frame', type=int, default=0, help='skip frame in reading dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--shuffle', default=False, action='store_true', help='suffle dataset')
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')

    parser.add_argument('--model_name', type=str, help='model_name')
    parser.add_argument('--load_path', type=str, default=None, help='load_path')

    training_args = parser.parse_args()

    return training_args
