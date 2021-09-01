from args.visualization_args import parse_visualization_args
from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_snapshot
import random
# from visualization.visualizer import Visualizer

if __name__ == '__main__':

    dataloader_args, model_args, load_path, is_testing, pred_frames_num, seq_index, gif_name = parse_visualization_args()
    dataloader = get_dataloader(dataloader_args)

    if load_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(load_path)
    elif model_args.model_name:
        model_args.pred_frames_num = pred_frames_num if is_testing else dataloader.dataset.future_frames_num
        model_args.keypoints_num = dataloader.dataset.keypoints_num
        model = get_model(model_args)
    else:
        raise Exception("Please provide either a model_name or a load_path to a trained model.")

    model.zero_grad()

    if seq_index is None:
        seq_index = random.randint(0, dataloader.dataset.__len__())
        print(seq_index)
    data = dataloader.dataset.__getitem__(seq_index, visualize=True)

    exit()
    obs_pose = None
    obs_mask = None
    obs_image_path = None

    future_mask = None
    future_pose = None
    future_image_path = None

    if not is_testing:
        if model.args.use_mask:
            future.append(data[-1])
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
        visualizer.visualizer_2D(poses=vis_poses, masks=vis_masks, name=gif_name)
    else:
        visualizer.visualizer_3D(poses=vis_poses, name=gif_name)
