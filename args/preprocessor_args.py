import argparse


def parse_preprocessor_args():
    args = __parse_preprocessor_args()

    return args


def __parse_preprocessor_args():
    parser = argparse.ArgumentParser('Preprocessor Arguments')
    parser.add_argument('--dataset_name', type=str,
                        choices=['somof_posetrack', 'posetrack', 'somof_3dpw', '3dpw', 'jta'], default='posetrack')
    parser.add_argument('--dataset_path', type=str, default='./raw_data', help='path of dataset')
    parser.add_argument('--data_usage', type=str, default='train', choices=['train', 'validation', 'test'])
    parser.add_argument('--use_mask', default=False, action='store_true', help='use visibility mask')
    parser.add_argument('--interactive', default=False, action='store_true', help='consider interaction')
    parser.add_argument('--output_name', type=str, help='name of preprocessed csv file')
    parser.add_argument('--obs_frame_num', type=int, default=16)
    parser.add_argument('--pred_frame_num', type=int, default=14)
    parser.add_argument('--skip_frame_num', type=int, default=1)
    parser.add_argument('--use_video_once', default=False, action='store_true')
    parser.add_argument('--3D', default=False, action='store_true', help='use if data is in 3D, default is 2D')

    preprocessor_args = parser.parse_args()
    preprocessor_args.device = 'cuda'

    return preprocessor_args
