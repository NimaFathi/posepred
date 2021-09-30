import logging

import hydra
from omegaconf import DictConfig

from configs.helper import DataloaderArgs, ModelArgs
from data_loader.my_dataloader import get_dataloader
from factory.evaluator import Evaluator
from models import get_model
from path_definition import HYDRA_PATH
from utils.reporter import Reporter
from utils.save_load import load_snapshot

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="evaluate")
def evaluate(cfg: DictConfig):
    dataloader_args = DataloaderArgs(cfg.dataloader.dataset_file_name, cfg.keypoint_dim, cfg.interactive,
                                     cfg.persons_num, cfg.use_mask, cfg.skip_num, cfg.dataloader.batch_size,
                                     cfg.dataloader.shuffle, cfg.pin_memory, cfg.num_workers)
    model_args = ModelArgs(cfg.model.model_name, cfg.use_mask, cfg.keypoint_dim)

    if cfg.train_dataset is not None:
        train_dataloader_args = DataloaderArgs(cfg.train_dataset, cfg.keypoint_dim, cfg.interactive, cfg.persons_num,
                                               cfg.use_mask, cfg.skip_num, 1024, False, cfg.pin_memory, cfg.num_workers)
    else:
        train_dataloader_args = None
    dataloader = get_dataloader(dataloader_args)
    reporter = Reporter(attrs=cfg.metrics.pose_metrics + cfg.metrics.mask_metrics)
    if cfg.load_path:
        model, _, _, _, _ = load_snapshot(cfg.load_path)
    elif model_args.model_name:
        model_args.pred_frames_num = dataloader.dataset.future_frames_num
        model_args.keypoints_num = dataloader.dataset.keypoints_num
        model = get_model(model_args).to('cuda')
        if model_args.model_name == 'nearest_neighbor':
            assert train_dataloader_args is not None, 'Please provide a train_dataset for nearest_neighbor model.'
            model.train_dataloader = get_dataloader(train_dataloader_args)
    else:
        msg = "Please provide either a model_name or a load_path to a trained model."
        logger.error(msg)
        raise Exception(msg)
    evaluator = Evaluator(model, dataloader, reporter, cfg.interactive, cfg.model.loss.loss_name,
                          cfg.metrics.pose_metrics,
                          cfg.metrics.mask_metrics, cfg.rounds_num)
    evaluator.evaluate()

    return dataloader_args, model_args, cfg.load_path, cfg.interactive, cfg.distance_loss, cfg.metrics.pose_metrics, cfg.metrics.mask_metrics, cfg.rounds_num, train_dataloader_args


if __name__ == '__main__':
    evaluate()
