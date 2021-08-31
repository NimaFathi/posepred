import sys
import time
import matplotlib.pyplot as plt
import json
import numpy as np
import torch

from utils.average_meter import AverageMeter


class Reporter:

    def __init__(self):
        self.attrs = dict()
        self.attrs['ADE'] = AverageMeter()
        self.attrs['FDE'] = AverageMeter()
        self.attrs['vel_loss'] = AverageMeter()
        self.attrs['mask_loss'] = AverageMeter()
        self.attrs['mask_acc'] = AverageMeter()
        self.start_time = None

        self.history = dict()
        self.history['ADE'] = []
        self.history['FDE'] = []
        self.history['vel_loss'] = []
        self.history['mask_loss'] = []
        self.history['mask_acc'] = []
        self.history['time'] = []

    def update(self, metrics, batch_size):
        for key, value in metrics.items():
            self.attrs.get(key).update(value, batch_size)

    def epoch_finished(self):
        self.history.get('time').append(time.time() - self.start_time)
        for key, avg_meter in self.attrs.items():
            self.history.get(key).append(avg_meter.get_average())
        self.reset_avr_meters()

    def reset_avr_meters(self):
        self.start_time = None
        for i, avg_meter in enumerate(self.attrs.values()):
            avg_meter.reset()

    def print_values(self, use_mask):
        msg = 'epoch:' + str(len(self.history['time']))
        for key, value in self.history.items():
            if not use_mask and 'mask' in key:
                continue
            msg += '| ' + key + ': %.2f' % value[-1].detach().cpu().numpy()
        print(msg)
        sys.stdout.flush()

    def save_plots(self, use_mask, save_dir):
        for key, value in self.history.items():
            if not use_mask and 'mask' in key:
                continue
            value = [v.detach().cpu().numpy() for v in value]
            with open(save_dir + '/plots/' + key + '.json', "w") as f:
                json.dump(value, f, indent=4)
            plt.plot(value)
            plt.xlabel('epoch')
            plt.ylabel(key)
            plt.savefig(save_dir + '/plots/' + key + '.png')

    def print_mean_std(self, use_mask):
        msg = ''
        for key, value in self.history.items():
            if not use_mask and 'mask' in key:
                continue
            if torch.is_tensor(value[0]):
                value = [v.detach().cpu().numpy() for v in value]
            msg += key + ': (mean=%.3f, std=%.3f)' % (np.mean(value), np.std(value)) + '\n'
        print(msg)
