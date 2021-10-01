import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from data_loader.my_dataloader import get_dataloader
from models import MODELS
from optimizers import OPTIMIZERS
from schedulers import SCHEDULERS
from factory.trainer import Trainer
from utils.reporter import Reporter
from utils.save_load import load_snapshot, save_snapshot, save_args, setup_training_dir

from path_definition import HYDRA_PATH
from path_definition import ROOT_DIR

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="train")
def train(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        raise Exception('either specify a load_path or config a model.')
    print(OmegaConf.to_yaml(cfg))
    exit()
    trainer_args = TrainerArgs(cfg.epochs, cfg.interactive, cfg.start_epoch, cfg.lr, cfg.decay_factor,
                               cfg.decay_patience, cfg.distance_loss, cfg.mask_loss_weight, cfg.snapshot_interval)
    model_args = ModelArgs(cfg.model.model_name, cfg.use_mask, cfg.keypoint_dim)

    train_dataloader = get_dataloader(cfg.train_dataset, cfg.data)
    valid_dataloader = get_dataloader(cfg.valid_dataset, cfg.data)

    if cfg.load_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(cfg.load_path)
        cfg.start_epoch = epoch
        cfg.save_dir = cfg.load_path[:cfg.load_path.rindex('snapshots/')]
    else:
        cfg.model.pred_frames_num = train_dataloader.dataset.future_frames_num
        cfg.model.keypoints_num = train_dataloader.dataset.keypoints_num
        model = MODELS[cfg.model.type](cfg.model)
        optimizer = OPTIMIZERS[cfg.optimizer.type](model.parameters(), cfg.optimizer)
        cfg.save_dir = setup_training_dir(ROOT_DIR)
        train_reporter = Reporter(state='train')
        valid_reporter = Reporter(state='valid')
        save_args({'trainer_args': trainer_args, 'model_args': model.args}, trainer_args.save_dir)
        save_snapshot(model, optimizer, trainer_args.lr, 0, train_reporter, valid_reporter, trainer_args.save_dir)

    model = model.to(cfg.device)
    scheduler = SCHEDULERS[cfg.scheduler.type](optimizer, cfg.scheduler)
    trainer = Trainer(trainer_args, model, train_dataloader, valid_dataloader, optimizer, scheduler, train_reporter,
                      valid_reporter)
    trainer.train()


if __name__ == '__main__':
    train()
