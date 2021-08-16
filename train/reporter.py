import time

from utils.average_meter import AverageMeter


class Reporter:

    def __init__(self):
        self.train_time = time.time()
        self.train = dict()
        self.train['vel_loss'] = AverageMeter()
        self.train['mask_loss'] = AverageMeter()
        self.train['mask_acc'] = AverageMeter()
        self.train['ADE'] = AverageMeter()
        self.train['FDE'] = AverageMeter()

        self.validation_time = time.time()
        self.validation = dict()
        self.validation['vel_loss'] = AverageMeter()
        self.validation['mask_loss'] = AverageMeter()
        self.validation['mask_acc'] = AverageMeter()
        self.validation['ADE'] = AverageMeter()
        self.validation['FDE'] = AverageMeter()

    def update_train(self, batch_size, values):
        self.train_time = time.time() - self.train_time
        for i, avg_meter in enumerate(self.train.values()):
            avg_meter.update(values[i], batch_size)

    def update_validation(self, batch_size, values):
        self.validation_time = time.time() - self.validation_time
        for i, avg_meter in enumerate(self.validation.values()):
            avg_meter.update(values[i], batch_size)

    def reset(self):
        self.train_time = time.time()
        for i, avg_meter in enumerate(self.train.values()):
            avg_meter.reset()
        self.validation_time = time.time()
        for i, avg_meter in enumerate(self.validation.values()):
            avg_meter.reset()

    def print_train_values(self):
        print("time:", self.train_time)
        for key, avg_meter in self.train.items():
            print(key + ":", avg_meter.get_average())
        print('-' * 30)

    def print_validation_values(self):
        print("time:", self.validation_time)
        for key, avg_meter in self.validation.items():
            print(key + ":", avg_meter.get_average())
        print('-' * 30)


reporter = Reporter()
reporter.print_train_values()
reporter.update_train(1, [1, 2, 3, 4, 5])
reporter.print_train_values()
reporter.update_train(2, [0, 2, 0, 4, 5])
reporter.print_train_values()
