import argparse

from args.helper import DataloaderArgs, ModelArgs


class VisualizationArgs:
    def __init__(self, dataset_name, keypoint_dim, model_name=None, load_path=None, gif_name=None, seq_index=None,
                 is_interactive=False, persons_num=1, is_testing=True, pred_frames_num=None, use_mask=False,
                 skip_frame=0):
        # dataloader_args
        self.dataset_name = dataset_name
        self.keypoint_dim = keypoint_dim
        self.is_interactive = is_interactive
        self.persons_num = persons_num
        self.is_testing = is_testing
        self.pred_frames_num = pred_frames_num
        self.use_mask = use_mask
        self.skip_frame = skip_frame
        self.batch_size = 1
        self.shuffle = False
        self.pin_memory = False
        self.num_workers = 0

        self.model_name = model_name
        self.load_path = load_path
        self.seq_index = seq_index
        self.gif_name = gif_name


def parse_visualization_args():
    args = __parse_visualization_args()
    dataloader_args = DataloaderArgs(args.dataset_name, args.keypoint_dim, args.is_interactive, args.persons_num,
                                     args.use_mask, args.is_testing, args.skip_frame, args.batch_size, args.shuffle,
                                     args.pin_memory, args.num_workers)
    model_args = ModelArgs(args.model_name, args.use_mask, args.keypoint_dim)

    return dataloader_args, model_args, args.load_path, args.is_testing, args.pred_frames_num, args.seq_index, args.gif_name


def __parse_visualization_args():
    parser = argparse.ArgumentParser('Visualization Arguments')

    # dataloader_args
    parser.add_argument('-dataset_name', type=str, help='test_dataset_name')
    parser.add_argument('-keypoint_dim', type=int, help='dimension of each keypoint')
    parser.add_argument('-is_interactive', type=bool, default=False, help='support interaction of people')
    parser.add_argument('-persons_num', type=int, default=1, help='number of people in each sequence')
    parser.add_argument('-is_testing', type=bool, default=True, help='provide ground truth if false')
    parser.add_argument('-pred_frames_num', type=int, default=None, help='number of future frames to predict')
    parser.add_argument('-use_mask', type=bool, default=False, help='visibility mask')
    parser.add_argument('-skip_frame', type=int, default=0, help='skip frame in reading dataset')

    parser.add_argument('-model_name', type=str, help='model_name')
    parser.add_argument('-load_path', type=str, default=None, help='load_path')
    parser.add_argument('-seq_index', type=int, default=None, help='index of a sequence in dataset.')
    parser.add_argument('-gif_name', type=str, default=None, help='name of generated gif')

    visualization_args = parser.parse_args()
    visualization_args.batch_size = 1
    visualization_args.shuffle = False
    visualization_args.pin_memory = False
    visualization_args.num_workers = 0

    return visualization_args
