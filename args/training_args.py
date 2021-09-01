import argparse

from args.helper import TrainerArgs, DataloaderArgs, ModelArgs


class TrainingArgs:
    def __init__(self, train_dataset_name, valid_dataset_name, model_name, keypoint_dim, epochs, start_epoch=0,
                 lr=0.001, decay_factor=0.95, decay_patience=20, distance_loss='L1', mask_loss_weight=0.25,
                 snapshot_interval=20, is_interactive=False, persons_num=1, use_mask=False, skip_frame=0,
                 batch_size=1, shuffle=True, pin_memory=False, num_workers=0, hidden_size=200, hardtanh_limit=10,
                 n_layers=1, dropout_enc=0, dropout_pose_dec=0, dropout_mask_dec=0, load_path=None):
        # trainer_args
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.lr = lr
        self.decay_factor = decay_factor
        self.decay_patience = decay_patience
        self.distance_loss = distance_loss
        self.mask_loss_weight = mask_loss_weight
        self.snapshot_interval = snapshot_interval

        # dataloader_args
        self.train_dataset_name = train_dataset_name
        self.valid_dataset_name = valid_dataset_name
        self.keypoint_dim = keypoint_dim
        self.is_interactive = is_interactive
        self.persons_num = persons_num
        self.use_mask = use_mask
        self.skip_frame = skip_frame
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.is_testing = False

        # model_args
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.hardtanh_limit = hardtanh_limit
        self.n_layers = n_layers
        self.dropout_enc = dropout_enc
        self.dropout_pose_dec = dropout_pose_dec
        self.dropout_mask_dec = dropout_mask_dec

        self.load_path = load_path


def parse_training_args():
    args = __parse_training_args()
    trainer_args = TrainerArgs(args.epochs, args.is_interactive, args.start_epoch, args.lr, args.decay_factor,
                               args.decay_patience, args.distance_loss, args.mask_loss_weight, args.snapshot_interval)
    train_dataloader_args = DataloaderArgs(args.train_dataset_name, args.keypoint_dim, args.is_interactive,
                                           args.persons_num, args.use_mask, args.is_testing, args.skip_frame,
                                           args.batch_size, args.shuffle, args.pin_memory, args.num_workers)
    valid_dataloader_args = DataloaderArgs(args.valid_dataset_name, args.keypoint_dim, args.is_interactive,
                                           args.persons_num, args.use_mask, args.is_testing, args.skip_frame,
                                           args.batch_size, args.shuffle, args.pin_memory, args.num_workers)
    model_args = ModelArgs(args.model_name, args.use_mask, args.keypoint_dim, args.hidden_size, args.hardtanh_limit,
                           args.n_layers, args.dropout_enc, args.dropout_pose_dec, args.dropout_mask_dec)

    return trainer_args, train_dataloader_args, valid_dataloader_args, model_args, args.load_path


def __parse_training_args():
    parser = argparse.ArgumentParser('Training Arguments')

    # trainer_args
    parser.add_argument('-epochs', type=int, help='number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-decay_factor', type=float, default=0.95, help='decay_factor for learning_rate')
    parser.add_argument('-decay_patience', type=int, default=20, help='decay_patience for learning_rate')
    parser.add_argument('-distance_loss', type=str, default='L1', help='use L1 or L2 as distance loss.')
    parser.add_argument('-mask_loss_weight', type=int, default=0.25, help='weight of mask-loss')
    parser.add_argument('-snapshot_interval', type=int, default=20, help='save snapshot every N epochs')

    # dataloader_args
    parser.add_argument('-train_dataset_name', type=str, help='train_dataset_name')
    parser.add_argument('-valid_dataset_name', type=str, help='valid_dataset_name')
    parser.add_argument('-keypoint_dim', type=int, help='dimension of each keypoint')
    parser.add_argument('-is_interactive', type=bool, default=False, help='support interaction of people')
    parser.add_argument('-persons_num', type=int, default=1, help='number of people in each sequence')
    parser.add_argument('-use_mask', type=bool, default=False, help='visibility mask')
    parser.add_argument('-skip_frame', type=int, default=0, help='skip frame in reading dataset')
    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-shuffle', type=bool, default=True)
    parser.add_argument('-pin_memory', type=bool, default=False)
    parser.add_argument('-num_workers', type=int, default=0, help='num_workers')

    # model_args
    parser.add_argument('-model_name', type=str, help='model_name')
    parser.add_argument('-hidden_size', type=int, default=200)
    parser.add_argument('-hardtanh_limit', type=float, default=10)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-dropout_enc', type=float, default=0)
    parser.add_argument('-dropout_pose_dec', type=float, default=0)
    parser.add_argument('-dropout_mask_dec', type=float, default=0)

    parser.add_argument('-load_path', type=str, default=None, help='load_snapshot_path')

    training_args = parser.parse_args()
    training_args.is_testing = False

    return training_args
