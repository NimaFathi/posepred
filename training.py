import torch.optim as optim

from consts import ROOT_DIR
from args.training_args import TrainingArgs
from args.helper import TrainerArgs, DataloaderArgs, ModelArgs
from data_loader.data_loader import get_dataloader
from utils.save_load import load_snapshot, get_model, save_snapshot, save_args, setup_training_dir
from utils.reporter import Reporter
from entangled.trainer import Trainer


if __name__ == '__main__':

    args = TrainingArgs(train_dataset_name='simple_dataset', valid_dataset_name='simple_dataset', model_name='lstm_vel',
                        keypoint_dim=2, epochs=6, load_path=None)

    trainer_args = TrainerArgs(args.epochs, args.is_interactive, args.start_epoch, args.lr, args.decay_factor,
                               args.decay_patience, args.distance_loss, args.mask_loss_weight, args.snapshot_interval)

    train_dataloader_args = DataloaderArgs(args.train_dataset_name, args.keypoint_dim, args.is_interactive,
                                           args.use_mask, args.is_testing, args.skip_frame, args.batch_size,
                                           args.shuffle, args.pin_memory, args.num_workers)
    valid_dataloader_args = DataloaderArgs(args.valid_dataset_name, args.keypoint_dim, args.is_interactive,
                                           args.use_mask, args.is_testing, args.skip_frame, args.batch_size,
                                           args.shuffle, args.pin_memory, args.num_workers)

    model_args = ModelArgs(args.model_name, args.use_mask, args.keypoint_dim, args.hidden_size, args.hardtanh_limit,
                           args.n_layers, args.dropout_enc, args.dropout_pose_dec, args.dropout_mask_dec)

    train_dataloader = get_dataloader(train_dataloader_args)
    valid_dataloader = get_dataloader(valid_dataloader_args)

    if args.load_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(args.load_path)
        trainer_args.start_epoch = epoch
        trainer_args.save_dir = args.load_path[:args.load_path.rindex('snapshots/')]
    else:
        model_args.pred_frames_num = train_dataloader.dataset.future_frames_num
        model_args.keypoints_num = train_dataloader.dataset.keypoints_num
        model = get_model(model_args)
        optimizer = optim.Adam(model.parameters(), lr=trainer_args.lr)
        train_reporter = Reporter()
        valid_reporter = Reporter()
        trainer_args.save_dir = setup_training_dir(ROOT_DIR)
        save_args({'trainer_args': trainer_args, 'model_args': model.args}, trainer_args.save_dir)
        save_snapshot(model, optimizer, trainer_args.lr, 0, train_reporter, valid_reporter, trainer_args.save_dir)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=trainer_args.decay_factor,
                                                     patience=trainer_args.decay_patience, threshold=1e-8, verbose=True)

    trainer = Trainer(trainer_args, model, train_dataloader, valid_dataloader, optimizer, scheduler, train_reporter,
                      valid_reporter)
    trainer.train()
