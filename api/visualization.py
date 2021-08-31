from args.visualization_args import parse_visualization_args
from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_snapshot
from visualization.visualizer import Visualizer

if __name__ == '__main__':

    dataloader_args, model_args, load_path, is_testing, seq_index = parse_visualization_args()
    dataloader = get_dataloader(dataloader_args)

    if load_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(load_path)
    elif model_args.model_name:
        model_args.pred_frames_num = dataloader.dataset.future_frames_num
        model_args.keypoints_num = dataloader.dataset.keypoints_num
        model = get_model(model_args)
    else:
        raise Exception("Please provide either a model_name or a load_path to a trained model.")

    model.zero_grad()
    data = dataloader.dataset.__getitem__(seq_index)

    vis_poses = []
    vis_masks = []

    if not is_testing:
        if model.args.use_mask:
            vis_masks.append(data[-1])
        else:
            obs_pose, obs_vel, target_pose, target_vel = data
        vis_poses.append(data[len(data) / 2])
        outputs = model(data[:len(data) / 2])
    else:
        outputs = model(data)

    vis_poses.append(data[0])
    vis_poses.append(outputs[0])
    if model.args.use_mask:
        vis_masks.append(data[2])
        vis_masks.append(outputs[1])

    visualizer = Visualizer(dataset=dataloader_args.dataset_name)
    if dataloader_args.keypoint_dim == 2:
        visualizer.visualizer_2D(poses=vis_poses, masks=vis_masks)
    else:
        visualizer.visualizer_3D(poses=vis_poses)
