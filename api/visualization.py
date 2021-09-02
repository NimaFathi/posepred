from args.visualization_args import parse_visualization_args
from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_snapshot
import random
import torch

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

    print('obs_pose:', data['obs_pose'].shape)
    print('future_pose:', data['future_pose'].shape)
    print('obs_image:', len(data['obs_image']))
    print('future_image:', len(data['future_image']))
    print('obs_cam_ex:', torch.tensor(data['obs_cam_ex']).shape)
    print('future_cam_ex:', torch.tensor(data['future_cam_ex']).shape)
    print('cam_in:', torch.tensor(data['cam_in']).shape)

    exit()

    #
    #
    # visualizer = Visualizer(dataset=dataloader_args.dataset_name)
    # if dataloader_args.keypoint_dim == 2:
    #     visualizer.visualizer_2D(poses=vis_poses, masks=vis_masks, name=model.args.model_name)
    # else:
    #     visualizer.visualizer_3D(poses=vis_poses, name=model.args.model_name)
