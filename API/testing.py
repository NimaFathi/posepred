from path_definition import ROOT_DIR
from args.testing_args import TestingArgs
from args.helper import DataloaderArgs, ModelArgs
from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_snapshot, save_args, setup_testing_dir
from entangled.tester import Tester

if __name__ == '__main__':

    args = TestingArgs(dataset_name='simple_non_interactive_dataset', model_name='zero_velocity', pred_frames_num=2,
                       load_path=None, keypoint_dim=2)

    dataloader_args = DataloaderArgs(args.dataset_name, args.keypoint_dim, args.is_interactive, args.use_mask,
                                     args.is_testing, args.skip_frame, args.batch_size, args.shuffle, args.pin_memory,
                                     args.num_workers)

    dataloader = get_dataloader(dataloader_args)

    if args.load_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(args.load_path)
    elif args.model_name:
        model_args = ModelArgs(args.model_name, args.use_mask, args.keypoint_dim)
        model_args.pred_frames_num = args.pred_frames_num
        model_args.keypoints_num = dataloader.dataset.keypoints_num
        model = get_model(model_args)
    else:
        raise Exception("Please provide either a model_name or a load_path to a trained model.")

    save_dir = setup_testing_dir(ROOT_DIR)
    save_args({'dataloader_args': dataloader_args, 'model_args': model.args}, save_dir)

    tester = Tester(model, dataloader, args.is_interactive, save_dir)
    tester.test()
