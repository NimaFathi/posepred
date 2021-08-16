import time

from utils.average_meter import AverageMeter


class Reporter:

    def __init__(self):
        train_time = time.time()

        train_vel_loss = AverageMeter()
        train_mask_loss = AverageMeter()
        train_mask_acc = AverageMeter()
        train_ADE = AverageMeter()
        train_FDE = AverageMeter()

        validation_vel_loss = AverageMeter()
        validation_mask_loss = AverageMeter()
        validation_mask_acc = AverageMeter()
        validation_ADE = AverageMeter()
        validation_FDE = AverageMeter()
