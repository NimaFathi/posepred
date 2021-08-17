import argparse


class TrainingArgs:
    def __init__(self, epochs, start_epoch=0, lr=0.001, decay_factor=0.95, decay_patience=20, distance_loss='L1',
                 mask_loss_weight=0.25, save_interval=20):
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.lr = lr
        self.decay_factor = decay_factor
        self.decay_patience = decay_patience
        self.distance_loss = distance_loss
        self.mask_loss_weight = mask_loss_weight
        self.save_interval = save_interval


def parse_training_args():
    parser = argparse.ArgumentParser('Arguments for Learning')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=0, default=200, help='start epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay_factor', type=float, default=0.95, help='decay_factor for learning_rate')
    parser.add_argument('--decay_patience', type=int, default=20, help='decay_patience for learning_rate')
    parser.add_argument('--distance_loss', type=str, default='L1', help='use L1 or L2 as distance loss.')
    parser.add_argument('--mask_loss_weight', type=int, default=0.25, help='weight of mask-loss')
    parser.add_argument('--save_interval', type=int, default=20, help='save model every N epochs')
    training_args = parser.parse_args()
    return training_args
