import argparse

from args.helper import DataloaderArgs, ModelArgs


def parse_evaluation_args():
    args = __parse_evaluation_args()
    dataloader_args = DataloaderArgs(args.dataset_name, args.keypoint_dim, args.is_interactive, args.persons_num,
                                     args.use_mask, args.skip_frame, args.batch_size, args.shuffle, args.pin_memory,
                                     args.num_workers)
    model_args = ModelArgs(args.model_name, args.use_mask, args.keypoint_dim)

    return dataloader_args, model_args, args.load_path, args.is_interactive, args.distance_loss


def __parse_evaluation_args():
    parser = argparse.ArgumentParser('Evaluation Arguments')

    # dataloader_args
    parser.add_argument('-dataset_name', type=str, help='test_dataset_name')
    parser.add_argument('-keypoint_dim', type=int, help='dimension of each keypoint')
    parser.add_argument('-is_interactive', type=bool, default=False, help='support interaction of people')
    parser.add_argument('-persons_num', type=int, default=1, help='number of people in each sequence')
    parser.add_argument('-use_mask', type=bool, default=False, help='visibility mask')
    parser.add_argument('-skip_frame', type=int, default=0, help='skip frame in reading dataset')
    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-shuffle', type=bool, default=True)
    parser.add_argument('-pin_memory', type=bool, default=False)
    parser.add_argument('-num_workers', type=int, default=0, help='num_workers')

    parser.add_argument('-model_name', type=str, help='model_name')
    parser.add_argument('-load_path', type=str, default=None, help='load_path')
    parser.add_argument('-distance_loss', type=str, default='L1', help='use L1 or L2 as distance loss.')

    evaluation_args = parser.parse_args()

    return evaluation_args
