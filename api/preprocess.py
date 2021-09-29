import logging
from logging import config

import hydra
from omegaconf import DictConfig

from path_definition import HYDRA_PATH
from path_definition import LOGGER_CONF
from preprocessor.dpw_preprocessor import Preprocessor3DPW
from preprocessor.human36m_preprocessor import PreprocessorHuman36m
from preprocessor.jaad_preprocessor import JAADPreprocessor
from preprocessor.jta_preprocessor import JTAPreprocessor
from preprocessor.pie_preprocessor import PIEPreprocessor
from preprocessor.posetrack_preprocessor import PoseTrackPreprocessor
from preprocessor.somof_3dpw_preprocessor import SoMoF3DPWPreprocessor
from preprocessor.somof_posetrack_preprocessor import SoMoFPoseTrackPreprocessor

config.fileConfig(LOGGER_CONF)
logger = logging.getLogger('consoleLogger')


@hydra.main(config_path=HYDRA_PATH, config_name="preprocess")
def preprocess(cfg: DictConfig):
    assert cfg.dataset_name in ['somof_posetrack', 'posetrack', 'somof_3dpw', '3dpw', 'jta', 'jaad', 'pie',
                                'human3.6m'], "invalid dataset name"
    assert cfg.data_type in ['train', 'validation', 'test'], "data_type must be in ['train', 'test', 'validation']"
    if cfg.keypoint_dim == 2:
        cfg.is_3D = False
    elif cfg.keypoint_dim == 3:
        cfg.is_3D = True
    else:
        msg = "Dimension of data must be either 2 or 3"
        logger.error(msg=msg)
        raise Exception(msg)
    if cfg.dataset_name == 'posetrack':
        preprocessor = PoseTrackPreprocessor(
            mask=cfg.use_mask, dataset_path=cfg.official_annotation_path,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=0, use_video_once=cfg.use_video_once)
    elif cfg.dataset_name == 'jta':
        preprocessor = JTAPreprocessor(
            is_3d=cfg.is_3D, dataset_path=cfg.official_annotation_path,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once
        )
    elif cfg.dataset_name == 'somof_posetrack':
        preprocessor = SoMoFPoseTrackPreprocessor(
            mask=cfg.use_mask, dataset_path=cfg.official_annotation_path,
            obs_frame_num=16, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=14, skip_frame_num=1, use_video_once=True
        )
    elif cfg.dataset_name == 'somof_3dpw':
        preprocessor = SoMoF3DPWPreprocessor(
            dataset_path=cfg.official_annotation_path,
            obs_frame_num=16, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=14, skip_frame_num=0, use_video_once=True
        )
    elif cfg.dataset_name == '3dpw':
        preprocessor = Preprocessor3DPW(
            dataset_path=cfg.official_annotation_path,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once
        )
    elif cfg.dataset_name == 'jaad':
        preprocessor = JAADPreprocessor(
            dataset_path=cfg.official_annotation_path, annotate=cfg.annotate, image_dir=cfg.image_dir,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once,
            annotation_path=cfg.joints_annotation_path
        )
    elif cfg.dataset_name == 'pie':
        preprocessor = PIEPreprocessor(
            dataset_path=cfg.official_annotation_path, annotate=cfg.annotate, image_dir=cfg.image_dir,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once,
            annotation_path=cfg.joints_annotation_path
        )
    elif cfg.dataset_name == 'human3.6m':
        preprocessor = PreprocessorHuman36m(
            dataset_path=cfg.official_annotation_path,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once
        )
    else:
        msg = "Invalid preprocessor."
        logger.error(msg)
        raise Exception(msg)
    preprocessor.normal(data_type=cfg.data_type)


if __name__ == '__main__':
    preprocess()
