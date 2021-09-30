import json
import logging
import os
import pickle

import torch
import torch.optim as optim

from api.helper import JSONEncoder_
from models import get_model

logger = logging.getLogger(__name__)


# TODO map_location="cuda:0" ???
def load_snapshot(load_snapshot_path):
    snapshot = torch.load(load_snapshot_path, map_location='cpu')
    model = get_model(snapshot['model_args']).to('cuda')
    model.load_state_dict(snapshot['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=snapshot['optimizer_lr'])
    return model, optimizer, snapshot['epoch'], snapshot['train_reporter'], snapshot['valid_reporter']


def save_snapshot(model, optimizer, optimizer_lr, epoch, train_reporter, valid_reporter, save_path):
    logger.info('### Taking Snapshot ###')
    snapshot = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_lr': optimizer_lr,
        'model_args': model.args,
        'epoch': epoch,
        'train_reporter': train_reporter,
        'valid_reporter': valid_reporter
    }
    torch.save(snapshot, os.path.join(save_path, 'snapshots', '%03d.pt' % epoch))
    del snapshot


def save_args(args, save_dir):
    for key, arg in args.items():
        with open(os.path.join(save_dir, key + '.txt'), 'w') as f:
            f.write(json.dumps(arg, indent=4, cls=JSONEncoder_))


def save_test_results(result_df, result_tensor, save_dir):
    result_df.to_csv(os.path.join(save_dir, 'outputs', 'results.csv'), index=False)
    with open(os.path.join(save_dir, 'outputs', 'results.pkl'), 'wb') as f:
        pickle.dump(result_tensor, f)


def setup_training_dir(root_dir):
    train_dir = os.path.join(root_dir, 'exps', 'train')
    os.makedirs(train_dir, exist_ok=True)
    for i in range(1, 1000000):
        new_dir = os.path.join(train_dir, str(i))
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir, exist_ok=False)
            os.makedirs(os.path.join(new_dir, 'snapshots'), exist_ok=False)
            os.makedirs(os.path.join(new_dir, 'plots'), exist_ok=False)
            os.makedirs(os.path.join(new_dir, 'metrics_history'), exist_ok=False)
            return new_dir
    msg = "Too many folders exist."
    logger.error(msg=msg)
    raise Exception(msg)


def setup_testing_dir(root_dir):
    test_dir = os.path.join(root_dir, 'exps', 'test')
    os.makedirs(test_dir, exist_ok=True)
    for i in range(1, 1000000):
        new_dir = os.path.join(test_dir, str(i))
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir, exist_ok=False)
            os.makedirs(os.path.join(new_dir, 'outputs'), exist_ok=False)
            return new_dir
    msg = "Too many folders exist."
    logger.error(msg=msg)
    raise Exception(msg)


def setup_visualization_dir(root_dir):
    vis_dir = os.path.join(root_dir, 'exps', 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    for i in range(1, 1000000):
        new_dir = os.path.join(vis_dir, str(i))
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir, exist_ok=False)
            return new_dir
    msg = "Too many folders exist."
    logger.error(msg=msg)
    raise Exception(msg)
