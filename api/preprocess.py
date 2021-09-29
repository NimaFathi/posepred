import logging
from logging import config

from args.preprocessing_args import parse_preprocessor_args
from path_definition import LOGGER_CONF
from preprocessor.dpw_preprocessor import Preprocessor3DPW
from preprocessor.jaad_preprocessor import JAADPreprocessor
from preprocessor.jta_preprocessor import JTAPreprocessor
from preprocessor.pie_preprocessor import PIEPreprocessor
from preprocessor.posetrack_preprocessor import PoseTrackPreprocessor
from preprocessor.somof_3dpw_preprocessor import SoMoF3DPWPreprocessor
from preprocessor.somof_posetrack_preprocessor import SoMoFPoseTrackPreprocessor
from preprocessor.human36m_preprocessor import PreprocessorHuman36m
config.fileConfig(LOGGER_CONF)
logger = logging.getLogger('consoleLogger')

if __name__ == '__main__':
    args = parse_preprocessor_args()

    if args.dataset_name == 'posetrack':
        preprocessor = PoseTrackPreprocessor(
            mask=args.use_mask, dataset_path=args.official_annotation_path,
            obs_frame_num=args.obs_frames_num, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=args.pred_frames_num, skip_frame_num=0, use_video_once=args.use_video_once)
    elif args.dataset_name == 'jta':
        preprocessor = JTAPreprocessor(
            is_3d=args.is_3D, dataset_path=args.official_annotation_path,
            obs_frame_num=args.obs_frames_num, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=args.pred_frames_num, skip_frame_num=args.skip_num, use_video_once=args.use_video_once
        )
    elif args.dataset_name == 'somof_posetrack':
        preprocessor = SoMoFPoseTrackPreprocessor(
            mask=args.use_mask, dataset_path=args.official_annotation_path,
            obs_frame_num=16, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=14, skip_frame_num=1, use_video_once=True
        )
    elif args.dataset_name == 'somof_3dpw':
        preprocessor = SoMoF3DPWPreprocessor(
            dataset_path=args.official_annotation_path,
            obs_frame_num=16, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=14, skip_frame_num=0, use_video_once=True
        )
    elif args.dataset_name == '3dpw':
        args.is_disentangle = False
        preprocessor = Preprocessor3DPW(
            dataset_path=args.official_annotation_path,
            obs_frame_num=args.obs_frames_num, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=args.pred_frames_num, skip_frame_num=args.skip_num, use_video_once=args.use_video_once
        )
    elif args.dataset_name == 'jaad':
        preprocessor = JAADPreprocessor(
            dataset_path=args.official_annotation_path, annotate=args.annotate, image_dir=args.image_dir,
            obs_frame_num=args.obs_frames_num, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=args.pred_frames_num, skip_frame_num=args.skip_num, use_video_once=args.use_video_once,
            annotation_path=args.joints_annotation_path
        )
    elif args.dataset_name == 'pie':
        preprocessor = PIEPreprocessor(
            dataset_path=args.official_annotation_path, annotate=args.annotate, image_dir=args.image_dir,
            obs_frame_num=args.obs_frames_num, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=args.pred_frames_num, skip_frame_num=args.skip_num, use_video_once=args.use_video_once,
            annotation_path=args.joints_annotation_path
        )
    elif args.dataset_name == 'human3.6m':
        preprocessor = PreprocessorHuman36m(
            dataset_path=args.official_annotation_path,
            obs_frame_num=args.obs_frames_num, custom_name=args.output_name, is_interactive=args.interactive,
            pred_frame_num=args.pred_frames_num, skip_frame_num=args.skip_num, use_video_once=args.use_video_once
        )
    else:
        msg = "Invalid preprocessor."
        logger.error(msg)
        raise Exception(msg)
    preprocessor.normal(data_type=args.data_usage)
