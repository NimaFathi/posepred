import torch.optim as optim


def adam(params, args):
    return optim.Adam(params, lr=args.lr)
