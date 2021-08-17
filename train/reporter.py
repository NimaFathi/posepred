import time

from utils.average_meter import AverageMeter


class Reporter:

    def __init__(self):
        self.attrs = dict()
        self.attrs['vel_loss'] = AverageMeter()
        self.attrs['mask_loss'] = AverageMeter()
        self.attrs['mask_acc'] = AverageMeter()
        self.attrs['ADE'] = AverageMeter()
        self.attrs['FDE'] = AverageMeter()
        self.time = time.time()

        self.history = dict()
        self.history['vel_loss'] = []
        self.history['mask_loss'] = []
        self.history['mask_acc'] = []
        self.history['ADE'] = []
        self.history['FDE'] = []

    def update(self, values, batch_size):
        self.time = time.time() - self.time
        for i, avg_meter in enumerate(self.attrs.values()):
            avg_meter.update(values[i], batch_size)

    def reset(self):
        self.time = time.time()
        for i, avg_meter in enumerate(self.attrs.values()):
            avg_meter.reset()

    def print_values(self):
        print("time:", self.time)
        for key, avg_meter in self.attrs.items():
            print(key + ":", avg_meter.get_average())
        print('-' * 30)

    def next_epoch(self):

