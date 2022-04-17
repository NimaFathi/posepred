import sys
import os
from tempfile import tempdir
import time
import json
from traceback import print_tb
from cv2 import sort
import matplotlib.pyplot as plt

import numpy as np
import torch
import pandas as pd

from utils.average_meter import AverageMeter


class Reporter:

    def __init__(self, state=''):
        self.state = state
        self.start_time = None
        self.attrs = None
        self.history = None

    def setup(self, attrs):
        self.attrs = {}
        for attr in attrs:
            self.attrs[attr] = AverageMeter()
        self.history = {}
        self.min_attrs = {}
        for attr in attrs:
            self.history[attr] = []
            self.min_attrs[attr] = float('inf')
        self.history['time'] = []

    def update(self, attrs, batch_size, dynamic=False, counts=None):
        if self.attrs is None or self.history is None:
            self.setup(attrs)
        for key, value in attrs.items():
            if dynamic:
                if key not in self.attrs.keys():
                    self.attrs[key] = AverageMeter()
                    self.history[key] = []
            if counts is not None and key in counts.keys():
                self.attrs.get(key).update(value, counts[key])
            else:
                self.attrs.get(key).update(value, batch_size)

    def epoch_finished(self, tb=None, mf=None):
        self.history.get('time').append(time.time() - self.start_time)
        for key, avg_meter in self.attrs.items():
            value = avg_meter.get_average()
            value = value.detach().cpu().numpy() if torch.is_tensor(value) else value
            self.history.get(key).append(float(value))

            if self.min_attrs[key] > value:
                self.min_attrs[key] = value

            if tb is not None:
                tb.add_scalar(self.state + '_' + key, float(value), len(self.history.get(key)))
            if mf is not None:
                mf.log_metric(self.state + '_' + key, float(value), len(self.history.get(key)))
                mf.log_metric(self.state + '_best_' + key, float(self.min_attrs.get(key)), len(self.history.get(key)))
        self.reset_avr_meters()

    def reset_avr_meters(self):
        self.start_time = None
        for i, avg_meter in enumerate(self.attrs.values()):
            avg_meter.reset()

    def print_values(self, logger, use_mask):
        msg = self.state + '-epoch' + str(len(self.history['time'])) + ': '
        for key, value in self.history.items():
            if not use_mask and 'mask' in key:
                continue
            msg += key + ': %.5f, ' % value[-1]
        logger.info(str(msg))
        sys.stdout.flush()

    def save_data(self, use_mask, save_dir):
        for key, value in self.history.items():
            if not use_mask and 'mask' in key:
                continue
            with open(os.path.join(save_dir, 'metrics_history', '_'.join((self.state, key)) + '.json'), "w") as f:
                json.dump(value, f, indent=4)

    def print_mean_std(self, logger, use_mask):
        for key, value in self.history.items():
            if not use_mask and 'mask' in key:
                continue
            logger.info(str(key) + ': (mean=%.5f, std=%.6f)' % (np.mean(value), np.std(value)))

    def print_pretty_metrics(self, logger, use_mask, metrics):
        actions = []
        for k in self.history.keys():
            if not use_mask and 'mask' in k:
                continue
            if metrics[0] in k:
                actions.append(k[len(metrics[0])+1:])
        actions = list(sorted(actions))
        logger.info(' |'.join(["actions".ljust(15)]+[a.center(15) for a in list(metrics)]))
        logger.info("_"*20*(len(list(metrics))+1))
        for action in actions:
            to_print = []
            for metric in list(metrics):
                to_print.append(np.mean(self.history.get(f'{metric}_{action}')))
            logger.info(' |'.join([action.ljust(15)]+ [str(np.around(a, 4)).center(15) for a in to_print]))
            
    def save_csv_metrics(self, use_mask, metrics, addr):
        actions = []
        for k in self.history.keys():
            if not use_mask and 'mask' in k:
                continue
            if metrics[0] in k:
                actions.append(k[len(metrics[0])+1:])
        actions = list(sorted(actions))
        out = pd.DataFrame(columns=["action"]+list(metrics))

        for action in actions:
            to_print = []
            out_dict = {}
            for metric in list(metrics):
                out_dict[metric] = [np.mean(self.history.get(f'{metric}_{action}'))]
            out_dict["action"] = action
            temp = [action]+ [a for a in to_print]
            # out=out.append(temp)
            df_temp = pd.DataFrame(out_dict)
            out = pd.concat([out, df_temp], ignore_index=True, axis = 0)
        # TODO: save csv file

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
