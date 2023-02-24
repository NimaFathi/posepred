import logging
import os
from itertools import chain

import hydra
from omegaconf import DictConfig

from uncertainty.main import load_dc_model
from data_loader import get_dataloader
from factory.trainer import Trainer
from losses import LOSSES
from models import MODELS
from optimizers import OPTIMIZERS
from path_definition import HYDRA_PATH
from schedulers import SCHEDULERS
from utils.reporter import Reporter
from utils.save_load import load_snapshot, save_snapshot, setup_training_dir
from factory.uncertainty_evaluator import UncertaintyEvaluator

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="train")
def train(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)

    train_dataloader = get_dataloader(cfg.train_dataset, cfg.data)
    cfg.data.is_testing = True
    valid_dataloader = get_dataloader(cfg.valid_dataset, cfg.data)

    if cfg.load_path is not None:
        model, loss_module, optimizer, optimizer_args, epoch, train_reporter, valid_reporter = load_snapshot(
            cfg.load_path)
        cfg.start_epoch = epoch
        cfg.optimizer = optimizer_args
        cfg.save_dir = cfg.load_path[:cfg.load_path.rindex('snapshots/')]
    else:
        #new commented
        # cfg.model.keypoints_num = train_dataloader.dataset.keypoints_num
        # cfg.model.mean_pose = train_dataloader.dataset.mean_pose
        # cfg.model.std_pose = train_dataloader.dataset.std_pose

        model = MODELS[cfg.model.type](cfg.model)
        loss_module = LOSSES[cfg.model.loss.type](cfg.model.loss)
        optimizer = OPTIMIZERS[cfg.optimizer.type](
            chain(model.parameters(), loss_module.parameters()), cfg.optimizer)
        
        train_reporter = Reporter(state='train')
        valid_reporter = Reporter(state='valid')
        cfg.save_dir = os.getcwd()
        setup_training_dir(cfg.save_dir)
        save_snapshot(model, loss_module, optimizer, cfg.optimizer,
                      0, train_reporter, valid_reporter, cfg.save_dir)
    uncertainty_evaluator = None
    if cfg.eval_uncertainty:
        dataset_name = 'Human36m'
        uncertainty_model = load_dc_model(dataset_name, cfg.oodu_load_path)
        train_uncertainty_evaluator = UncertaintyEvaluator(cfg, train_dataloader, model, uncertainty_model,
                                                     cfg.model.obs_frames_num, cfg.model.pred_frames_num,
                                                     dataset_name, train_reporter, in_line=True)
        validation_uncertainty_evaluator = UncertaintyEvaluator(cfg, valid_dataloader, model, uncertainty_model,
                                                     cfg.model.obs_frames_num, cfg.model.pred_frames_num,
                                                     dataset_name, valid_reporter, in_line=True)
    scheduler = SCHEDULERS[cfg.scheduler.type](optimizer, cfg.scheduler)
    trainer = Trainer(cfg, train_dataloader, valid_dataloader, model, loss_module, optimizer, cfg.optimizer, scheduler,
                      train_reporter, valid_reporter, train_uncertainty_evaluator, validation_uncertainty_evaluator)
    trainer.train()


if __name__ == '__main__':
    train()
