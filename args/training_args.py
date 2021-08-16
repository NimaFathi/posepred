import argparse


class TrainingArgs:
    def __init__(self, epochs, start_epoch=0, lr=0.001, decay_factor=0.95, decay_patience=20,
                 save_checkpoint_interval=20):
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.lr = lr
        self.decay_factor = decay_factor
        self.decay_patience = decay_patience
        self.save_checkpoint_interval = save_checkpoint_interval


def parse_training_args():
    parser = argparse.ArgumentParser('Arguments for Learning')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=0, default=200, help='start epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay_factor', type=float, default=0.95, help='decay_factor for learning_rate')
    parser.add_argument('--decay_patience', type=int, default=20, help='decay_patience for learning_rate')
    parser.add_argument('--save_checkpoint_interval', type=int, default=20)
    training_args = parser.parse_args()
    return training_args
