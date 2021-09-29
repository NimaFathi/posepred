import argparse
import logging
from logging import config

from path_definition import LOGGER_CONF

config.fileConfig(LOGGER_CONF)
logger = logging.getLogger('consoleLogger')


def parse_preprocessor_args():
    args = __parse_preprocessor_args()
    if args.keypoint_dim == 2:
        args.is_3D = False
    elif args.keypoint_dim == 3:
        args.is_3D = True
    else:
        msg = "Dimension of data must be either 2 or 3"
        logger.error(msg=msg)
        raise Exception(msg)

    return args


def __parse_preprocessor_args():
    parser = argparse.ArgumentParser('Preprocessor Arguments')
    parser.add_argument(
        '--dataset_name', type=str,
        choices=['somof_posetrack', 'posetrack', 'somof_3dpw', '3dpw', 'jta', 'jaad', 'pie', 'human3.6m'],
        help='dataset_name'
    )
    parser.add_argument('--official_annotation_path', type=str, default='./raw_data', help='path of dataset')
    parser.add_argument('--keypoint_dim', type=int, choices=[2, 3], help='dimension of each keypoint')
    parser.add_argument('--data_usage', type=str, choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--use_mask', default=False, action='store_true', help='use visibility mask')
    parser.add_argument('--interactive', default=False, action='store_true', help='consider interaction')
    parser.add_argument('--output_name', type=str, help='name of preprocessed csv file')
    parser.add_argument('--obs_frames_num', type=int, help='number of frames to observe', default=16)
    parser.add_argument('--pred_frames_num', type=int, help='number of frames to predict', default=14)
    parser.add_argument('--skip_num', type=int, default=0, help='number of frames to skip')
    parser.add_argument('--use_video_once', default=False, action='store_true')
    parser.add_argument(
        '--annotate', default=False, action='store_true',
        help='implies if dataset contains annotations or we have to generate annotations with openpifpaf'
             'if this is True we will annotate data upon <image_dir> with openpifpaf library'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        help='use images to annotate with openpifpaf if <annotate> is True'
    )
    parser.add_argument(
        '--joints_annotation_path',
        type=str,
        help='joints annotation path if necessary for dataset its mandatory for datasets like, JAAD and PIE'
    )
    preprocessor_args = parser.parse_args()
    preprocessor_args.device = 'cuda'

    return preprocessor_args
