import logging
from logging import config

import torch.optim as optim

from args.training_args import parse_training_args
from data_loader.data_loader import get_dataloader
from factory.trainer import Trainer
from path_definition import LOGGER_CONF
from path_definition import ROOT_DIR
from utils.reporter import Reporter
from utils.save_load import load_snapshot, get_model, save_snapshot, save_args, setup_training_dir

config.fileConfig(LOGGER_CONF)
logger = logging.getLogger('consoleLogger')

if __name__ == '__main__':

    trainer_args, train_dataloader_args, valid_dataloader_args, model_args, load_path = parse_training_args()
    train_dataloader = get_dataloader(train_dataloader_args)
    valid_dataloader = get_dataloader(valid_dataloader_args) if valid_dataloader_args.dataset_name is not None else None

    if load_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(load_path)
        trainer_args.start_epoch = epoch
        trainer_args.save_dir = load_path[:load_path.rindex('snapshots/')]
    else:
        model_args.pred_frames_num = train_dataloader.dataset.future_frames_num
        model_args.keypoints_num = train_dataloader.dataset.keypoints_num
        model = get_model(model_args)
        optimizer = optim.Adam(model.parameters(), lr=trainer_args.lr)
        train_reporter = Reporter(state='train')
        valid_reporter = Reporter(state='valid')
        trainer_args.save_dir = setup_training_dir(ROOT_DIR)
        save_args({'trainer_args': trainer_args, 'model_args': model.args}, trainer_args.save_dir)
        save_snapshot(model, optimizer, trainer_args.lr, 0, train_reporter, valid_reporter, trainer_args.save_dir)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=trainer_args.decay_factor,
                                                     patience=trainer_args.decay_patience, threshold=1e-8, verbose=True)

    trainer = Trainer(trainer_args, model, train_dataloader, valid_dataloader, optimizer, scheduler, train_reporter,
                      valid_reporter)
    trainer.train()
