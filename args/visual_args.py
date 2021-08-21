import argparse


class VisualArgs:
    def __init__(self, dataset_name, keypoint_dim, pred_frames_num=None, model_name=None, load_path=None, seq_index=None, is_interactive=False, is_testing=True, use_mask=False,
                 skip_frame=0):
        # dataloader_args
        self.dataset_name = dataset_name
        self.keypoint_dim = keypoint_dim
        self.is_interactive = is_interactive
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.skip_frame = skip_frame
        self.batch_size = 1
        self.shuffle = False
        self.pin_memory = False
        self.num_workers = 0

        self.model_name = model_name
        self.load_path = load_path

        self.pred_frames_num = pred_frames_num
        self.seq_index = seq_index


def parse_visualization_args():
    parser = argparse.ArgumentParser('Visualization Arguments')

    # dataloader_args
    parser.add_argument('--dataset_name', type=str, help='test_dataset_name')
    parser.add_argument('--keypoint_dim', type=int, help='dimension of each keypoint')
    parser.add_argument('--is_interactive', type=bool, default=False, help='support interaction of people')
    parser.add_argument('--is_testing', type=bool, default=True, help='provide ground truth if false')
    parser.add_argument('--use_mask', type=bool, default=False, help='visibility mask')
    parser.add_argument('--skip_frame', type=int, default=0, help='skip frame in reading dataset')

    parser.add_argument('--model_name', type=str, help='model_name')
    parser.add_argument('--load_path', type=str, default=None, help='load_path')
    parser.add_argument('--seq_index', type=int, default=None, help='index of a sequence in dataset.')
    parser.add_argument('--pred_frames_num', type=str, default=None, help='number of frames to predict')

    visualization_args = parser.parse_args()
    return visualization_args
