import logging
import hydra
from omegaconf import DictConfig

from data_loader.my_dataloader import get_dataloader
from models import MODELS
from losses import LOSSES
from factory.evaluator import Evaluator
from path_definition import HYDRA_PATH
from utils.reporter import Reporter
from utils.save_load import load_snapshot

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="evaluate")
def evaluate(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)
    if cfg.train_dataset is not None:
        pass

    eval_dataloader = get_dataloader(cfg.eval_dataset, cfg.data)
    eval_reporter = Reporter(state='eval')
    if cfg.load_path is not None:
        model, loss_module, _, _, _, _, _ = load_snapshot(cfg.load_path)
    else:
        cfg.model.keypoint_dim = cfg.data.keypoint_dim
        cfg.model.pred_frames_num = eval_dataloader.dataset.future_frames_num
        cfg.model.keypoints_num = eval_dataloader.dataset.keypoints_num
        cfg.model.use_mask = cfg.data.use_mask
        model = MODELS[cfg.model.type](cfg.model)
        loss_module = LOSSES[cfg.loss.type](cfg.loss)
        if cfg.model.type == 'nearest_neighbor':
            assert cfg.train_dataset is not None, 'Please provide a train_dataset for nearest_neighbor model.'
            model.train_dataloader = get_dataloader(cfg.train_dataset, cfg.data)

    evaluator = Evaluator(cfg, eval_dataloader, model, loss_module, eval_reporter)
    evaluator.evaluate()


if __name__ == '__main__':
    evaluate()
