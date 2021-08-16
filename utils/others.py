import torch
from models.lstm_vel.lstm_vel import LSTMVel3dpw


def get_model(args):
    if args.load_model_checkpoint is not None:
        model = load_model()
    if args.model_name == 'lstm_vel':
        return LSTMVel3dpw(args).to('cuda')


def load_model(opt, load_ckpt=None):
    if load_ckpt:
        ckpt = torch.load(load_ckpt, map_location='cpu')
    else:
        ckpt = torch.load(opt.load_ckpt, map_location='cpu')

    ckpt_opt = ckpt['opt']
    for key, val in ckpt_opt.__dict__.items():
        setattr(opt, key, val)
    model = set_model(opt)
    state_dict = ckpt['model']
    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
