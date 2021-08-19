from args.validation_args import ValidationArgs
from args.helper import DataloaderArgs
from data_loader.data_loader import get_dataloader
from utils.save_load import load_snapshot
from utils.reporter import Reporter
from entangled.validator import Validator

if __name__ == '__main__':

    args = ValidationArgs(dataset_name='simple_dataset', model_path='', keypoint_dim=2)

    dataloader_args = DataloaderArgs(args.dataset_name, args.keypoint_dim, args.is_interactive, args.use_mask,
                                     args.is_testing, args.skip_frame, args.batch_size, args.shuffle, args.pin_memory,
                                     args.num_workers)
    if args.model_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(args.model_path)
    else:
        raise Exception("No model is selected.")

    dataloader = get_dataloader(dataloader_args)
    reporter = Reporter()

    validator = Validator(model, dataloader, reporter, args.is_interactive, args.distance_loss)
    validator.validate()
