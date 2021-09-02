from args.visualization_args import parse_visualization_args
from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_snapshot
import random

# from visualization.visualizer import Visualizer

if __name__ == '__main__':

    dataloader_args, model_args, load_path, ground_truth, pred_frames_num, index = parse_visualization_args()
    dataloader = get_dataloader(dataloader_args)

    if load_path:
        model, optimizer, epoch, train_reporter, valid_reporter = load_snapshot(load_path)
    elif model_args.model_name:
        model_args.pred_frames_num = dataloader.dataset.future_frames_num if ground_truth else pred_frames_num
        assert model_args.pred_frames_num is not None, 'specify pred_frames_num'
        model_args.keypoints_num = dataloader.dataset.keypoints_num
        model = get_model(model_args)
    else:
        raise Exception("Please provide either a model_name or a load_path to a trained model.")

    model.zero_grad()

    index = random.randint(0, dataloader.dataset.__len__()) if index is None else index
    data = dataloader.dataset.__getitem__(index)

    # ['obs_pose', 'future_pose', 'obs_image', 'future_image', 'obs_cam_ex', 'future_cam_ex', 'cam_in']

    # for k, v in data.items():
    #     print(k, v)
    #     print('-' * 20)

    print('obs_pose:', data['obs_pose'].shape)
    print('future_pose:', data['future_pose'].shape)
    print('obs_image:', len(data['obs_image']))
    print('future_image:', len(data['future_image']))
    print('obs_cam_ex:', type(data['obs_cam_ex']))
    print('future_cam_ex:', type(data['future_cam_ex']))
    print('cam_in:', type(data['cam_in']))
    print(data.keys())

    exit()

    # obs_pose = None
    # obs_mask = None
    # obs_image_path = None
    #
    # future_mask = None
    # future_pose = None
    # future_image_path = None
    #
    # if not is_testing:
    #     if model.args.use_mask:
    #         future.append(data[-1])
    #     else:
    #         obs_pose, obs_vel, target_pose, target_vel = data
    #     vis_poses.append(data[len(data) / 2])
    #     outputs = model(data[:len(data) / 2])
    # else:
    #     outputs = model(data)
    #
    # vis_poses.append(data[0])
    # vis_poses.append(outputs[0])
    # if model.args.use_mask:
    #     vis_masks.append(data[2])
    #     vis_masks.append(outputs[1])
    #
    # visualizer = Visualizer(dataset=dataloader_args.dataset_name)
    # if dataloader_args.keypoint_dim == 2:
    #     visualizer.visualizer_2D(poses=vis_poses, masks=vis_masks, name=model.args.model_name)
    # else:
    #     visualizer.visualizer_3D(poses=vis_poses, name=model.args.model_name)
