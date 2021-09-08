import sys
import os
import time
import matplotlib.pyplot as plt
import json
import numpy as np
import torch

from utils.average_meter import AverageMeter


class Reporter:

    def __init__(self, state=''):
        self.state = state

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
            value = avg_meter.get_average()
            value = value.detach().cpu().numpy() if torch.is_tensor(value) else value
            self.history.get(key).append(float(value))
        self.reset_avr_meters()

    def reset_avr_meters(self):
        self.start_time = None
        for i, avg_meter in enumerate(self.attrs.values()):
            avg_meter.reset()

    def print_values(self, use_mask, end='\n'):
        msg = self.state + '-epoch' + str(len(self.history['time'])) + ': '
        for key, value in self.history.items():
            if not use_mask and 'mask' in key:
                continue
            msg += key + ': %.3f, ' % value[-1]
        print(msg, end=end)
        sys.stdout.flush()

    def save_data(self, use_mask, save_dir):
        for key, value in self.history.items():
            if not use_mask and 'mask' in key:
                continue
            with open(os.path.join(save_dir, 'data', '_'.join((self.state, key)) + '.json'), "w") as f:
                json.dump(value, f, indent=4)

    def print_mean_std(self, use_mask):
        for key, value in self.history.items():
            if not use_mask and 'mask' in key:
                continue
            print(key + ': (mean=%.3f, std=%.3f)' % (np.mean(value), np.std(value)))

    @staticmethod
    def save_plots(use_mask, save_dir, train_history, validiation_history, use_validation):
        for key, value in train_history.items():
            if not use_mask and 'mask' in key:
                continue
            X = list(range(1, len(value) + 1))
            plt.plot(X, value, color='b', label='_'.join(('train', key)))
            if use_validation and key in validiation_history.keys():
                plt.plot(X, validiation_history.get(key), color='g', label='_'.join(('validation', key)))
            plt.xlabel('epoch')
            plt.ylabel(key)
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'plots', key + '.png'))
            plt.close()
