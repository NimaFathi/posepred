import os

import torch
import torch.optim as optim

from models.lstm_vel.lstm_vel import LSTMVel3dpw


# TODO: edit get_model to support all models.
def get_model(model_args):
    if model_args.model_name == 'lstm_vel':
        return LSTMVel3dpw(model_args).to(torch.device('cuda'))


# TODO map_location="cuda:0" ???
def load_snapshot(load_snapshot_path, optimizer_lr):
    snapshot = torch.load(load_snapshot_path, map_location='cpu')
    model_args = snapshot['model_args']
    model = get_model(model_args).load_state_dict(snapshot['model_state'])
    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)
    return model, model_args, optimizer, snapshot['epoch']


def save_snapshot(model, optimizer, model_args, epoch, save_path):
    print('### Taking Snapshot ###')
    snapshot = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_args': model_args,
        'epoch': epoch,
    }
    torch.save(snapshot, os.path.join(save_path, '%03d.pt' % epoch))
    del snapshot


def create_new_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for i in range(1, 1000000):
        new_dir_path = os.path.join(dir_path, str(i) + '/')
        if not os.path.isdir(new_dir_path):
            os.makedirs(new_dir_path, exist_ok=False)
            os.makedirs(new_dir_path + 'snapshots/', exist_ok=False)
            return new_dir_path
    assert "Too many folders exist."
