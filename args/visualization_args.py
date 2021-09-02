import argparse

from args.helper import DataloaderArgs, ModelArgs


def parse_visualization_args():
    args = __parse_visualization_args()
    dataloader_args = DataloaderArgs(args.dataset_name, args.keypoint_dim, args.interactive, args.persons_num,
                                     args.use_mask, args.skip_frame, args.batch_size, args.shuffle, args.pin_memory,
                                     args.num_workers, is_testing=not args.ground_truth, is_visualizing=True)
    model_args = ModelArgs(args.model_name, args.use_mask, args.keypoint_dim)

    return dataloader_args, model_args, args.load_path, args.ground_truth, args.pred_frames_num, args.index


def __parse_visualization_args():
    parser = argparse.ArgumentParser('Visualization Arguments')

    # dataloader_args
    parser.add_argument('--dataset_name', type=str, help='test_dataset_name')
    parser.add_argument('--keypoint_dim', type=int, help='dimension of each keypoint')
    parser.add_argument('--interactive', default=False, action='store_true', help='consider interaction')
    parser.add_argument('--persons_num', type=int, default=1, help='number of people in each sequence')
    parser.add_argument('--pred_frames_num', type=int, default=None, help='number of future frames to predict')
    parser.add_argument('--use_mask', default=False, action='store_true', help='use visibility mask')
    parser.add_argument('--skip_num', type=int, default=0, help='number of frames to skip in reading dataset')

    parser.add_argument('--model_name', type=str, help='model_name')
    parser.add_argument('--load_path', type=str, default=None, help='load_path')
    parser.add_argument('--index', type=int, default=None, help='index of a sequence in dataset')
    parser.add_argument('--ground_truth', default=False, action='store_true', help='use ground-truth future frames')

    visualization_args = parser.parse_args()
    visualization_args.batch_size = 1
    visualization_args.shuffle = False
    visualization_args.pin_memory = False
    visualization_args.num_workers = 0

    return visualization_args
