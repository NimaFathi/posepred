import logging

import hydra
from omegaconf import DictConfig

from path_definition import HYDRA_PATH
from preprocessor.dpw_preprocessor import Preprocessor3DPW
from preprocessor.human36m_preprocessor import Human36mPreprocessor
from preprocessor.amass_preprocessor import AmassPreprocessor
from data_loader import DATASETS, DATA_TYPES

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="preprocess")
def preprocess(cfg: DictConfig):
    assert cfg.dataset in DATASETS, "invalid dataset name"
    assert cfg.data_type in DATA_TYPES, "data_type choices: " + str(DATA_TYPES)

    if cfg.keypoint_dim not in [2, 3]:
        msg = "Dimension of data must be either 2 or 3"
        logger.error(msg=msg)
        raise Exception(msg)
    
    if cfg.dataset == 'human3.6m':
        preprocessor = Human36mPreprocessor(
            dataset_path=cfg.official_annotation_path,
            custom_name=cfg.output_name, is_interactive=cfg.interactive,
            skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once,
            obs_frame_num=cfg.obs_frames_num, pred_frame_num= cfg.pred_frames_num,
            save_total_frames=cfg.save_total_frames
        )
    elif cfg.dataset == '3dpw':
        preprocessor = Preprocessor3DPW(
            dataset_path=cfg.official_annotation_path,
            obs_frame_num=cfg.obs_frames_num, custom_name=cfg.output_name, is_interactive=cfg.interactive,
            pred_frame_num=cfg.pred_frames_num, skip_frame_num=cfg.skip_num, use_video_once=cfg.use_video_once,
            save_total_frames=cfg.save_total_frames, load_60Hz=cfg.load_60Hz
        )
    elif cfg.dataset == 'amass':
        preprocessor = AmassPreprocessor(
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