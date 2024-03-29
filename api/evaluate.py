import logging
import hydra
from omegaconf import DictConfig

from data_loader import get_dataloader
from models import MODELS
from losses import LOSSES
from factory.evaluator import Evaluator
from utils.reporter import Reporter
from utils.save_load import load_snapshot
import os 

from path_definition import HYDRA_PATH

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="evaluate")
def evaluate(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)
    dataloader = get_dataloader(cfg.dataset, cfg.data)
    eval_reporter = Reporter(state='')
    if cfg.load_path is not None:
        model, loss_module, _, _, _, _, _ = load_snapshot(cfg.load_path)
        cfg.save_dir = cfg.load_path[:cfg.load_path.rindex('snapshots/')]
    else:
        cfg.model.pred_frames_num = dataloader.dataset.future_frames_num
        cfg.model.keypoints_num = dataloader.dataset.keypoints_num
        cfg.model.obs_frames_num = dataloader.dataset.obs_frames_num
        cfg.model.mean_pose = dataloader.dataset.mean_pose
        cfg.model.std_pose = dataloader.dataset.std_pose
        cfg.save_dir = os.getcwd()

        model = MODELS[cfg.model.type](cfg.model)
        loss_module = LOSSES[cfg.model.loss.type](cfg.model.loss)
        if cfg.model.type == 'nearest_neighbor':
            model.train_dataloader = get_dataloader(cfg.model.train_dataset, cfg.data)

    evaluator = Evaluator(cfg, dataloader, model, loss_module, eval_reporter)
    evaluator.evaluate()


if __name__ == '__main__':
    evaluate()
