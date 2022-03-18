import logging

import hydra
from omegaconf import DictConfig

from path_definition import HYDRA_PATH
from preprocessor.dpw_preprocessor import Preprocessor3DPW
from preprocessor.human36m_preprocessor import PreprocessorHuman36m
from preprocessor.jaad_preprocessor import JAADPreprocessor
from preprocessor.jta_preprocessor import JTAPreprocessor
from preprocessor.pie_preprocessor import PIEPreprocessor
from preprocessor.posetrack_preprocessor import PoseTrackPreprocessor
from preprocessor.somof_3dpw_preprocessor import SoMoF3DPWPreprocessor
from preprocessor.somof_posetrack_preprocessor import SoMoFPoseTrackPreprocessor
from preprocessor.stanford_preprocessor import StanfordPreprocessor
from data_loader import DATASETS, DATA_TYPES

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="preprocess")
def preprocess(cfg: DictConfig):
    assert cfg.dataset in DATASETS, "invalid dataset name"
    assert cfg.data_type in DATA_TYPES, "data_type choices: " + str(DATA_TYPES)
    if cfg.keypoint_dim == 2:
        is_3D = False
    elif cfg.keypoint_dim == 3:
        is_3D = True
    else:
        msg = "Dimension of data must be either 2 or 3"
        logger.error(msg=msg)
        raise Exception(msg)
    if cfg.dataset == 'posetrack':
        preprocessor = PoseTrackPreprocessor(
            dataset_path=cfg.official_annotation_path,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=0, use_video_once=cfg.use_video_once,
            save_total_frames=cfg.save_total_frames
        )
    elif cfg.dataset == 'jta':
        preprocessor = JTAPreprocessor(
            is_3d=is_3D, dataset_path=cfg.official_annotation_path,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once,
            save_total_frames=cfg.save_total_frames
        )
    elif cfg.dataset == 'stanford3.6m':
        preprocessor = StanfordPreprocessor(
            dataset_path=cfg.official_annotation_path,
            custom_name=cfg.output_name, is_interactive=cfg.interactive,
            skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once,
            obs_frame_num=cfg.obs_frames_num, pred_frame_num= cfg.pred_frames_num,
            save_total_frames=cfg.save_total_frames
        )
    elif cfg.dataset == 'somof_posetrack':
        preprocessor = SoMoFPoseTrackPreprocessor(
            dataset_path=cfg.official_annotation_path,
            obs_frame_num=16, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=14, skip_frame_num=1, use_video_once=True,
            save_total_frames=cfg.save_total_frames
        )
    elif cfg.dataset == 'somof_3dpw':
        preprocessor = SoMoF3DPWPreprocessor(
            dataset_path=cfg.official_annotation_path,
            obs_frame_num=16, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=14, skip_frame_num=0, use_video_once=True,
            save_total_frames=cfg.save_total_frames
        )
    elif cfg.dataset == '3dpw':
        preprocessor = Preprocessor3DPW(
            dataset_path=cfg.official_annotation_path,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once,
            save_total_frames=cfg.save_total_frames
        )
    elif cfg.dataset == 'jaad':
        preprocessor = JAADPreprocessor(
            dataset_path=cfg.official_annotation_path, annotate=cfg.annotate, image_dir=cfg.image_dir,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once,
            annotation_path=cfg.joints_annotation_path,
            save_total_frames=cfg.save_total_frames
        )
    elif cfg.dataset == 'pie':
        preprocessor = PIEPreprocessor(
            dataset_path=cfg.official_annotation_path, annotate=cfg.annotate, image_dir=cfg.image_dir,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once,
            annotation_path=cfg.joints_annotation_path,
            save_total_frames=cfg.save_total_frames
        )
    elif cfg.dataset == 'human3.6m':
        preprocessor = PreprocessorHuman36m(
            dataset_path=cfg.official_annotation_path,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once,
            save_total_frames=cfg.save_total_frames
        )
    else:
        msg = "Invalid preprocessor."
        logger.error(msg)
        raise Exception(msg)
    preprocessor.normal(data_type=cfg.data_type)


if __name__ == '__main__':
    preprocess()