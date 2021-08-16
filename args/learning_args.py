import argparse


class LearningArgs:
    def __init__(self, epochs, start_epoch=0, lr=0.001, lr_decay=0.95, save_checkpoint_interval=20):
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.lr = lr
        self.lr_decay = lr_decay
        self.save_checkpoint_interval = save_checkpoint_interval


def parse_learning_args():
    parser = argparse.ArgumentParser('Arguments for Learning')

    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=0, default=200, help='start epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='decay rate for learning rate')
    parser.add_argument('--save_checkpoint_interval', type=int, default=20)
    learnin_args = parser.parse_args()
    return learnin_args
