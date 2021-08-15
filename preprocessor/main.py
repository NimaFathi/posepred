import argparse

from jta_preprocessor import JTAPreprocessor
from posetrack_preprocessor import PoseTrackPreprocessor


def parse_option():
    parser = argparse.ArgumentParser('argument for predictions')
    parser.add_argument('--dataset', type=str, choices=['posetrack', '3dpw', 'jta'], default='posetrack')
    parser.add_argument('--dataset_path', type=str, default='./raw_data', help='path of dataset')
    parser.add_argument('--is_disentangle', type=bool, default=False)
    parser.add_argument('--obs_frame_num', type=int, default=16)
    parser.add_argument('--pred_frame_num', type=int, default=14)
    parser.add_argument('--skip_frame_num', type=int, default=1)
    parser.add_argument('--use_video_once', type=bool, default=False)
    parser.add_argument('--mask', type=bool, default=False)
    parser.add_argument('--is_3d', type=bool, default=False)

    opt = parser.parse_args()
    opt.device = 'cuda'

    return opt


if __name__ == '__main__':
    opt = parse_option()
    if opt.dataset == 'posetrack':
        preprocessor = PoseTrackPreprocessor(
            mask=opt.mask, dataset_path=opt.dataset_path, is_disentangle=opt.is_disentangle,
            obs_frame_num=opt.obs_frame_num,
            pred_frame_num=opt.pred_frame_num, skip_frame_num=opt.skip_frame_num, use_video_once=opt.use_video_once)
    elif opt.dataset == 'jta':
        preprocessor = JTAPreprocessor(
            is_3d=opt.is_3d, mask=opt.mask, dataset_path=opt.dataset_path, is_disentangle=opt.is_disentangle,
            obs_frame_num=opt.obs_frame_num,
            pred_frame_num=opt.pred_frame_num, skip_frame_num=opt.skip_frame_num, use_video_once=opt.use_video_once
        )
    else:
        preprocessor = None
    if opt.is_disentangle is True:
        preprocessor.disentangle()
    else:
        preprocessor.normal()
