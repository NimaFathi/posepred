import logging

import hydra
import torch.optim as optim
from omegaconf import DictConfig

from api.helper import TrainerArgs, DataloaderArgs, ModelArgs
from data_loader.my_dataloader import get_dataloader
from factory.trainer import Trainer
from models import get_model
from path_definition import HYDRA_PATH
from path_definition import ROOT_DIR
from utils.reporter import Reporter
from utils.save_load import load_snapshot, save_snapshot, save_args, setup_training_dir

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="train")
def train(cfg: DictConfig):
    if cfg.snapshot_interval == -1:
        cfg.snapshot_interval = cfg.epochs
    trainer_args = TrainerArgs(cfg.epochs, cfg.interactive, cfg.start_epoch, cfg.lr, cfg.decay_factor,
                               cfg.decay_patience, cfg.distance_loss, cfg.mask_loss_weight, cfg.snapshot_interval)
    train_dataloader_args = DataloaderArgs(cfg.train_dataset, cfg.keypoint_dim, cfg.interactive, cfg.persons_num,
                                           cfg.use_mask, cfg.skip_num, cfg.dataloader.batch_size,
                                           cfg.dataloader.shuffle, cfg.pin_memory, cfg.num_workers)
    valid_dataloader_args = DataloaderArgs(cfg.valid_dataset, cfg.keypoint_dim, cfg.interactive, cfg.persons_num,
                                           cfg.use_mask, cfg.skip_num, cfg.dataloader.batch_size,
                                           cfg.dataloader.shuffle, cfg.pin_memory, cfg.num_workers)
    model_args = ModelArgs(cfg.model.model_name, cfg.use_mask, cfg.keypoint_dim, cfg.model.hidden_size,
                           cfg.model.hardtanh_limit, cfg.model.n_layers,
                           cfg.model.dropout_enc, cfg.model.dropout_pose_dec, cfg.model.dropout_mask_dec)

    train_dataloader = get_dataloader(train_dataloader_args)
    valid_dataloader = get_dataloader(valid_dataloader_args) if valid_dataloader_args.dataset_name is not None else None

    if cfg.load_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(cfg.load_path)
        trainer_args.start_epoch = epoch
        trainer_args.save_dir = cfg.load_path[:cfg.load_path.rindex('snapshots/')]
    else:
        model_args.pred_frames_num = train_dataloader.dataset.future_frames_num
        model_args.keypoints_num = train_dataloader.dataset.keypoints_num
        model = get_model(model_args).to('cuda')
        optimizer = optim.Adam(model.parameters(), lr=trainer_args.lr)
        trainer_args.save_dir = setup_training_dir(ROOT_DIR)
        report_attrs = trainer_args.pose_metrics + trainer_args.mask_metrics
        train_reporter = Reporter(attrs=report_attrs, state='train')
        valid_reporter = Reporter(attrs=report_attrs, state='valid')
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
