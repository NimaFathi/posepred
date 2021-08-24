from args.visual_args import VisualArgs
from args.helper import DataloaderArgs, ModelArgs
from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_snapshot
from visualization.visualizer import Visualizer

if __name__ == '__main__':

    args = VisualArgs(dataset_name='simple_non_interactive_dataset', model_name='zero_velocity', load_path=None,
                      pred_frames_num=2, keypoint_dim=2, seq_index=5)

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

    model.zero_grad()
    data = dataloader.dataset.__getitem__(args.seq_index)

    vis_inputs = dict()
    if not args.is_testing:
        if model.args.use_mask:
            vis_inputs['target_mask'] = data[-1]
        else:
            obs_pose, obs_vel, target_pose, target_vel = data
        vis_inputs['target_pose'] = data[len(data) / 2]
        outputs = model(data[:len(data) / 2])
    else:
        outputs = model(data)

    vis_inputs['obs_pose'] = data[0]
    vis_inputs['pred_pose'] = outputs[0]
    if model.args.use_mask:
        vis_inputs['obs_mask'] = data[2]
        vis_inputs['pred_mask'] = outputs[1]

    visualizer = Visualizer(dataset_name=args.dataset_name)
    visualizer.visualize_2D(vis_inputs, )
