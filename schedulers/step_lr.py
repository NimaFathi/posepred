import torch.optim as optim


def step_lr(optimizer, args):
    return optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
