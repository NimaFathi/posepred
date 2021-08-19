from consts import ROOT_DIR
from args.testing_args import TestingArgs
from args.helper import DataloaderArgs
from data_loader.data_loader import get_dataloader
from utils.save_load import load_snapshot, save_args, setup_testing_dir
from entangled.tester import Tester

if __name__ == '__main__':

    args = TestingArgs(dataset_name='simple_dataset', model_path='', keypoint_dim=2)

    dataloader_args = DataloaderArgs(args.dataset_name, args.keypoint_dim, args.is_interactive, args.use_mask,
                                     args.is_testing, args.skip_frame, args.batch_size, args.shuffle, args.pin_memory,
                                     args.num_workers)

    if args.model_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(args.model_path)
    else:
        raise Exception("No model is selected.")

    dataloader = get_dataloader(dataloader_args)
    save_dir = setup_testing_dir(ROOT_DIR)
    save_args({'dataloader_args': dataloader_args, 'model_args': model.args}, save_dir)

    tester = Tester(model, dataloader, args.is_interactive, save_dir)
    tester.test()
