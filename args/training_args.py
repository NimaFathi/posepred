import argparse

from args.helper import TrainerArgs, DataloaderArgs, ModelArgs


def parse_training_args():
    args = __parse_training_args()
    if args.snapshot_interval == -1:
        args.snapshot_interval = args.epochs
    trainer_args = TrainerArgs(args.epochs, args.interactive, args.start_epoch, args.lr, args.decay_factor,
                               args.decay_patience, args.distance_loss, args.mask_loss_weight, args.snapshot_interval)
    train_dataloader_args = DataloaderArgs(args.train_dataset_name, args.keypoint_dim, args.interactive,
                                           args.persons_num, args.use_mask, args.skip_frame, args.batch_size,
                                           args.shuffle, args.pin_memory, args.num_workers)
    valid_dataloader_args = DataloaderArgs(args.valid_dataset_name, args.keypoint_dim, args.interactive,
                                           args.persons_num, args.use_mask, args.skip_frame, args.batch_size,
                                           args.shuffle, args.pin_memory, args.num_workers)
    model_args = ModelArgs(args.model_name, args.use_mask, args.keypoint_dim, args.hidden_size, args.hardtanh_limit,
                           args.n_layers, args.dropout_enc, args.dropout_pose_dec, args.dropout_mask_dec)

    return trainer_args, train_dataloader_args, valid_dataloader_args, model_args, args.load_path


def __parse_training_args():
    parser = argparse.ArgumentParser('Training Arguments')

    # trainer_args
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay_factor', type=float, default=0.95, help='decay_factor for learning_rate')
    parser.add_argument('--decay_patience', type=int, default=20, help='decay_patience for learning_rate')
    parser.add_argument('--distance_loss', type=str, default='L1', help='use L1 or L2 as distance loss.')
    parser.add_argument('--mask_loss_weight', type=int, default=0.25, help='weight of mask-loss')
    parser.add_argument('--snapshot_interval', type=int, default=-1, help='save snapshot every N epochs')

    # dataloader_args
    parser.add_argument('--train_dataset_name', type=str, help='train_dataset_name')
    parser.add_argument('--valid_dataset_name', type=str, help='valid_dataset_name')
    parser.add_argument('--keypoint_dim', type=int, help='dimension of each keypoint')
    parser.add_argument('--interactive', default=False, action='store_true', help='consider interaction')
    parser.add_argument('--persons_num', type=int, default=1, help='number of people in each sequence')
    parser.add_argument('--use_mask', default=False, action='store_true', help='use visibility mask')
    parser.add_argument('--skip_frame', type=int, default=0, help='skip frame in reading dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle dataset')
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')

    # model_args
    parser.add_argument('--model_name', type=str, help='model_name')
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--hardtanh_limit', type=float, default=10)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout_enc', type=float, default=0)
    parser.add_argument('--dropout_pose_dec', type=float, default=0)
    parser.add_argument('--dropout_mask_dec', type=float, default=0)

    parser.add_argument('--load_path', type=str, default=None, help='load_snapshot_path')

    training_args = parser.parse_args()

    return training_args
