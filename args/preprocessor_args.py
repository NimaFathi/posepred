import argparse


def parse_preprocessor_args():
    args = __parse_preprocessor_args()

    return args


def __parse_preprocessor_args():
    parser = argparse.ArgumentParser('arguments for make preprocessed static files')
    parser.add_argument(
        '--dataset', type=str,
        choices=['somof_posetrack', 'posetrack', 'somof_3dpw', '3dpw', 'jta'],
        default='posetrack'
    )
    parser.add_argument('--dataset_path', type=str, default='./raw_data', help='path of dataset')
    parser.add_argument('--mask', default=False, action='store_true')
    parser.add_argument('--data_type', type=str, default='train', choices=['train', 'validation', 'test'])
    parser.add_argument('--custom_name', type=str)
    parser.add_argument('--is_interactive', default=False, action='store_true')
    parser.add_argument('--obs_frame_num', type=int, default=16)
    parser.add_argument('--pred_frame_num', type=int, default=14)
    parser.add_argument('--skip_frame_num', type=int, default=1)
    parser.add_argument('--use_video_once', default=False, action='store_true')
    parser.add_argument('--is_3d', default=False, action='store_true')

    opt = parser.parse_args()
    opt.device = 'cuda'

    return opt
