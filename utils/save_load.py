import os
import json

import torch
import torch.optim as optim

from consts import ROOT_DIR
from args.helper import JSONEncoder_
from models.lstm_vel import LSTMVel


# TODO: support models with mask.
def get_model(model_args):
    if model_args.model_name == 'lstm_vel':
        return LSTMVel(model_args).to(torch.device('cuda'))


# TODO map_location="cuda:0" ???
def load_snapshot(load_snapshot_path):
    snapshot = torch.load(load_snapshot_path, map_location='cpu')
    model = get_model(snapshot['model_args']).load_state_dict(snapshot['model_state'])
    optimizer = optim.Adam(model.parameters(), lr=snapshot['optimizer_lr'])
    reporters = (snapshot['train_reporter'], snapshot['valid_reporter'])
    return model, optimizer, snapshot['epoch'], reporters


def save_snapshot(model, optimizer, optimizer_lr, epoch, train_reporter, valid_reporter, save_path):
    print('### Taking Snapshot ###')
    snapshot = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_lr': optimizer_lr,
        'model_args': model.args,
        'epoch': epoch,
        'train_reporter': train_reporter,
        'valid_reporter': valid_reporter
    }
    torch.save(snapshot, os.path.join(save_path + 'snapshots/', '%03d.pt' % epoch))
    del snapshot


def save_args(trainer_args, model_args, save_dir):
    with open(save_dir + 'trainer_args.txt', 'w') as f:
        f.write(json.dumps(trainer_args, indent=4, cls=JSONEncoder_))
    with open(save_dir + 'model_args.txt', 'w') as f:
        f.write(json.dumps(model_args, indent=4, cls=JSONEncoder_))


def create_save_dir():
    dir_path = create_new_dir(os.path.join(ROOT_DIR, 'exps/train/'))
    return dir_path


def create_new_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for i in range(1, 1000000):
        new_dir_path = os.path.join(dir_path, str(i) + '/')
        if not os.path.isdir(new_dir_path):
            os.makedirs(new_dir_path, exist_ok=False)
            os.makedirs(new_dir_path + 'snapshots/', exist_ok=False)
            os.makedirs(new_dir_path + 'plots/', exist_ok=False)
            return new_dir_path
    assert "Too many folders exist."
