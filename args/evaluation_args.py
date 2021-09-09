import argparse

from args.helper import DataloaderArgs, ModelArgs


def parse_evaluation_args():
    args = __parse_evaluation_args()
    dataloader_args = DataloaderArgs(args.dataset, args.keypoint_dim, args.interactive, args.persons_num,
                                     args.use_mask, args.skip_num, args.batch_size, args.shuffle, args.pin_memory,
                                     args.num_workers)
    model_args = ModelArgs(args.model, args.use_mask, args.keypoint_dim)

    if args.train_dataset is not None:
        train_dataloader_args = DataloaderArgs(args.train_dataset, args.keypoint_dim, args.interactive,
                                               args.persons_num, args.use_mask, args.skip_num, 1024, False,
                                               args.pin_memory, args.num_workers)
    else:
        train_dataloader_args = None

    return dataloader_args, model_args, args.load_path, args.interactive, args.distance_loss, args.rounds_num, train_dataloader_args


def __parse_evaluation_args():
    parser = argparse.ArgumentParser('Evaluation Arguments')

    # dataloader_args
    parser.add_argument('--dataset', type=str, help='dataset_name')
    parser.add_argument('--keypoint_dim', type=int, help='dimension of each keypoint')
    parser.add_argument('--interactive', default=False, action='store_true', help='consider interaction')
    parser.add_argument('--persons_num', type=int, default=1, help='number of people in each sequence')
    parser.add_argument('--use_mask', default=False, action='store_true', help='use visibility mask')
    parser.add_argument('--skip_num', type=int, default=0, help='number of frames to skip in reading dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle dataset')
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')

    parser.add_argument('--model', type=str, help='model_name')
    parser.add_argument('--load_path', type=str, help='load_path to trained model')
    parser.add_argument('--distance_loss', type=str, default='L1', help='use L1 or L2 as distance loss.')
    parser.add_argument('--rounds_num', type=int, default=1, help='number of rounds to evaluate.')

    parser.add_argument('--train_dataset', type=str, default=None, help='train dataset for nearest_neighbor model.')

    evaluation_args = parser.parse_args()

    return evaluation_args
