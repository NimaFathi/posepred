from args.visual_args import VisualArgs
from args.helper import DataloaderArgs, ModelArgs
from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_snapshot

if __name__ == '__main__':

    args = VisualArgs(dataset_name='simple_non_interactive_dataset', model_name='zero_velocity', load_path=None,
                      keypoint_dim=2, seq_index=5)

    dataloader_args = DataloaderArgs(args.dataset_name, args.keypoint_dim, args.is_interactive, args.use_mask,
                                     args.is_testing, args.skip_frame, args.batch_size, args.shuffle, args.pin_memory,
                                     args.num_workers)

    dataloader = get_dataloader(dataloader_args)

    if args.load_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(args.load_path)
    elif args.model_name:
        model_args = ModelArgs(args.model_name, args.use_mask, args.keypoint_dim)
        model_args.pred_frames_num = args.pred_frames_num if args.is_testing else dataloader.dataset.future_frames_num
        model_args.keypoints_num = dataloader.dataset.keypoints_num
        model = get_model(model_args)
    else:
        raise Exception("Please provide either a model_name or a load_path to a trained model.")

    print(dataloader.dataset.__getitem__(args.seq_index))
    # evaluator = Evaluator(model, dataloader, reporter, args.is_interactive, args.distance_loss)
    # evaluator.evaluate()
