from args.preprocessing_args import parse_preprocessor_args
from preprocessor.dpw_preprocessor import Preprocessor3DPW
from preprocessor.jta_preprocessor import JTAPreprocessor
from preprocessor.posetrack_preprocessor import PoseTrackPreprocessor
from preprocessor.somof_3dpw_preprocessor import SoMoF3DPWPreprocessor
from preprocessor.somof_posetrack_preprocessor import SoMoFPoseTrackPreprocessor

if __name__ == '__main__':
    args = parse_preprocessor_args()

    if args.dataset == 'posetrack':
        preprocessor = PoseTrackPreprocessor(
            mask=args.use_mask, dataset_path=args.dataset_path,
            obs_frame_num=16, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=14, skip_frame_num=1, use_video_once=True)
    elif args.dataset == 'jta':
        preprocessor = JTAPreprocessor(
            is_3d=args.is_3D, dataset_path=args.dataset_path,
            obs_frame_num=args.obs_frame_num, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=args.pred_frame_num, skip_frame_num=args.skip_num, use_video_once=args.use_video_once
        )
    elif args.dataset == 'somof_posetrack':
        preprocessor = SoMoFPoseTrackPreprocessor(
            mask=args.use_mask, dataset_path=args.dataset_path,
            obs_frame_num=16, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=14, skip_frame_num=1, use_video_once=True
        )
    elif args.dataset == 'somof_3dpw':
        preprocessor = SoMoF3DPWPreprocessor(
            dataset_path=args.dataset_path,
            obs_frame_num=16, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=14, skip_frame_num=1, use_video_once=True
        )
    elif args.dataset == '3dpw':
        args.is_disentangle = False
        preprocessor = Preprocessor3DPW(
            dataset_path=args.dataset_path,
            obs_frame_num=args.obs_frame_num, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=args.pred_frame_num, skip_frame_num=args.skip_num, use_video_once=args.use_video_once
        )
    else:
        raise Exception("Invalid preprocessor.")
    preprocessor.normal(data_type=args.data_type)
