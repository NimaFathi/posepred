
import sys
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
        self.start_time = None

        self.history = dict()
        self.history['vel_loss'] = []
        self.history['mask_loss'] = []
        self.history['mask_acc'] = []
        self.history['ADE'] = []
        self.history['FDE'] = []
        self.history['time'] = []

    def update(self, values, batch_size):
        for i, avg_meter in enumerate(self.attrs.values()):
            avg_meter.update(values[i], batch_size)

    def epoch_finished(self):
        self.history.get('time').append(time.time() - self.start_time)
        for key, avg_meter in self.attrs.items():
            self.history.get(key).append(avg_meter.get_average())
        self.reset_avr_meters()

    def reset_avr_meters(self):
        self.start_time = None
        for i, avg_meter in enumerate(self.attrs.values()):
            avg_meter.reset()

    def print_values(self):
        msg = 'epoch:' + str(len(self.history['time']))
        for key, item in self.history.items():
            msg += '| ' + key + ': %.2f' % item[-1]
        print(msg)
        sys.stdout.flush()
