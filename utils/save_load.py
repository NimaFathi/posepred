import os

import torch
from models.lstm_vel.lstm_vel import LSTMVel3dpw


# TODO: rewrite save and load

# TODO: edit get_model to support all models.
def get_model(model_args):
    if model_args.model_name == 'lstm_vel':
        return LSTMVel3dpw(model_args).to('cuda')


def load_model(load_path):
    checkpoint = torch.load(load_path, map_location='cpu')
    model_args = checkpoint['model_args']
    model = get_model(model_args).load_state_dict(checkpoint['model_state'])
    return model


def load_checkpoint(load_path):
    checkpoint = torch.load(load_path, map_location='cpu')
    model_args = checkpoint['model_args']
    model = get_model(model_args).load_state_dict(checkpoint['model_state'])

    return model, checkpoint['model_args'], checkpoint['dataloader_args'], checkpoint['training_args']


def save_checkpoint(model, optimizer, model_args, dataloader_args, training_args, epoch, save_path):
    print('==> Saving...')
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'model_args': model_args,
        'dataloader_args': dataloader_args,
        'training_args': training_args,
        'epoch': epoch,
    }
    torch.save(checkpoint, save_path)
    del checkpoint


def create_new_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for i in range(1, 1000000):
        new_dir_path = os.path.join(dir_path, str(i) + '/')
        if not os.path.isdir(new_dir_path):
            os.makedirs(new_dir_path, exist_ok=False)
            return new_dir_path
    assert "Too many folders exist."




