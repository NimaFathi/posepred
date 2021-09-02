from args.preprocessor_args import parse_preprocessor_args
from preprocessor.dpw_preprocessor import Preprocessor3DPW
from preprocessor.jta_preprocessor import JTAPreprocessor
from preprocessor.posetrack_preprocessor import PoseTrackPreprocessor
from preprocessor.somof_3dpw_preprocessor import SoMoF3DPWPreprocessor
from preprocessor.somof_posetrack_preprocessor import SoMoFPoseTrackPreprocessor

if __name__ == '__main__':
    opt = parse_preprocessor_args()

    if opt.dataset == 'posetrack':
        preprocessor = PoseTrackPreprocessor(
            mask=opt.mask, dataset_path=opt.dataset_path,
            obs_frame_num=16, custom_name=opt.custom_name, is_interactive=opt.is_interactive,
            pred_frame_num=14, skip_frame_num=1, use_video_once=True)
    elif opt.dataset == 'jta':
        preprocessor = JTAPreprocessor(
            is_3d=opt.is_3d, dataset_path=opt.dataset_path,
            obs_frame_num=opt.obs_frame_num, custom_name=opt.custom_name, is_interactive=opt.is_interactive,
            pred_frame_num=opt.pred_frame_num, skip_frame_num=opt.skip_frame_num, use_video_once=opt.use_video_once
        )
    elif opt.dataset == 'somof_posetrack':
        preprocessor = SoMoFPoseTrackPreprocessor(
            mask=opt.mask, dataset_path=opt.dataset_path,
            obs_frame_num=16, custom_name=opt.custom_name, is_interactive=opt.is_interactive,
            pred_frame_num=14, skip_frame_num=1, use_video_once=True
        )
    elif opt.dataset == 'somof_3dpw':
        preprocessor = SoMoF3DPWPreprocessor(
            dataset_path=opt.dataset_path,
            obs_frame_num=16, custom_name=opt.custom_name, is_interactive=opt.is_interactive,
            pred_frame_num=14, skip_frame_num=1, use_video_once=True
        )
    elif opt.dataset == '3dpw':
        opt.is_disentangle = False
        preprocessor = Preprocessor3DPW(
            dataset_path=opt.dataset_path,
            obs_frame_num=opt.obs_frame_num, custom_name=opt.custom_name, is_interactive=opt.is_interactive,
            pred_frame_num=opt.pred_frame_num, skip_frame_num=opt.skip_frame_num, use_video_once=opt.use_video_once
        )
    else:
        preprocessor = None
    preprocessor.normal(data_type=opt.data_type)
