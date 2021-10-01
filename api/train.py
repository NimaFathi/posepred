import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim

from configs.helper import TrainerArgs, DataloaderArgs, ModelArgs
from data_loader.my_dataloader import get_dataloader
from factory.trainer import Trainer
from models import get_model
from optimizers import OPTIMIZERS
from schedulers import S
from utils.reporter import Reporter
from utils.save_load import load_snapshot, save_snapshot, save_args, setup_training_dir
from path_definition import HYDRA_PATH
from path_definition import ROOT_DIR

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="train")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    exit()
    trainer_args = TrainerArgs(cfg.epochs, cfg.interactive, cfg.start_epoch, cfg.lr, cfg.decay_factor,
                               cfg.decay_patience, cfg.distance_loss, cfg.mask_loss_weight, cfg.snapshot_interval)
    model_args = ModelArgs(cfg.model.model_name, cfg.use_mask, cfg.keypoint_dim)

    train_dataloader = get_dataloader(cfg.train_dataset, cfg.data)
    valid_dataloader = get_dataloader(cfg.valid_dataset, cfg.data)

    if cfg.load_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(cfg.load_path)
        trainer_args.start_epoch = epoch
        trainer_args.save_dir = cfg.load_path[:cfg.load_path.rindex('snapshots/')]
    else:
        model_args.pred_frames_num = train_dataloader.dataset.future_frames_num
        model_args.keypoints_num = train_dataloader.dataset.keypoints_num
        model = get_model(model_args).to('cuda')
        optimizer = OPTIMIZERS[optimizer_type](model.parameters(), optimizer_args)
        trainer_args.save_dir = setup_training_dir(ROOT_DIR)
        train_reporter = Reporter(state='train')
        valid_reporter = Reporter(state='valid')
        save_args({'trainer_args': trainer_args, 'model_args': model.args}, trainer_args.save_dir)
        save_snapshot(model, optimizer, trainer_args.lr, 0, train_reporter, valid_reporter, trainer_args.save_dir)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=trainer_args.decay_factor,
                                                     patience=trainer_args.decay_patience, threshold=1e-8, verbose=True)
    trainer = Trainer(trainer_args, model, train_dataloader, valid_dataloader, optimizer, scheduler, train_reporter,
                      valid_reporter)
    trainer.train()
    return trainer_args, train_dataloader_args, valid_dataloader_args, model_args, cfg.load_path


if __name__ == '__main__':
    train()
