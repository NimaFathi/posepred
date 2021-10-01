import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from data_loader.my_dataloader import get_dataloader
from models import MODELS
from optimizers import OPTIMIZERS
from schedulers import SCHEDULERS
from factory.trainer import Trainer
from utils.reporter import Reporter
from utils.save_load import load_snapshot, save_snapshot

from path_definition import HYDRA_PATH

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="train")
def train(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        raise Exception('either specify a load_path or config a model.')
    print(OmegaConf.to_yaml(cfg))
    # model_args = ModelArgs(cfg.model.model_name, cfg.use_mask, cfg.keypoint_dim)

    train_dataloader = get_dataloader(cfg.train_dataset, cfg.data)
    valid_dataloader = get_dataloader(cfg.valid_dataset, cfg.data)

    if cfg.load_path:
        model, optimizer, optimizer_args, epoch, train_reporter, valid_reporter = load_snapshot(cfg.load_path)
        cfg.start_epoch = epoch
        cfg.optimizer = optimizer_args
        cfg.save_dir = cfg.load_path[:cfg.load_path.rindex('snapshots/')]
    else:
        cfg.model.keypoint_dim = cfg.data.keypoint_dim
        cfg.model.keypoint_dim = cfg.data.use_mask
        cfg.model.pred_frames_num = train_dataloader.dataset.future_frames_num
        cfg.model.keypoints_num = train_dataloader.dataset.keypoints_num
        model = MODELS[cfg.model.type](cfg.model)
        optimizer = OPTIMIZERS[cfg.optimizer.type](model.parameters(), cfg.optimizer)
        cfg.save_dir = os.getcwd()
        train_reporter = Reporter(state='train')
        valid_reporter = Reporter(state='valid')
        save_snapshot(model, optimizer, cfg.optimizer, 0, train_reporter, valid_reporter, cfg.save_dir)

    scheduler = SCHEDULERS[cfg.scheduler.type](optimizer, cfg.scheduler)

    args = {'epochs': cfg.epochs, 'start_epoch': cfg.start_epoch, 'is_interactive': cfg.data.is_interactive,
            'snapshot_interval': cfg.snapshot_interval, 'save_dir': cfg.save_dir, 'device': cfg.device}
    trainer = Trainer(args, train_dataloader, valid_dataloader, model, optimizer, cfg.optimizer, scheduler,
                      train_reporter, valid_reporter)
    trainer.train()


if __name__ == '__main__':
    train()
