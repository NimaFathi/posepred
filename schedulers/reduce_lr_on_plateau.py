import torch.optim as optim


def reduce_lr_on_plateau(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.mode, factor=args.factor, patience=args.patience,
                                                threshold=args.threshold, verbose=args.verbose)
