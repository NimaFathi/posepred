import logging
import os
import random
from logging import config

import torch

from args.visualization_args import parse_visualization_args
from data_loader.my_dataloader import get_dataloader
from path_definition import LOGGER_CONF
from path_definition import ROOT_DIR
from utils.others import pose_from_vel
from utils.save_load import load_snapshot
from visualization.visualizer import Visualizer
from models import get_model

config.fileConfig(LOGGER_CONF)
logger = logging.getLogger('consoleLogger')

if __name__ == '__main__':

    dataloader_args, model_args, load_path, ground_truth, pred_frames_num, index, images_dir, dataset_name = parse_visualization_args()
    dataloader = get_dataloader(dataloader_args)

    if load_path:
        model, _, _, _, _ = load_snapshot(load_path)
    elif model_args.model_name:
        model_args.pred_frames_num = dataloader.dataset.future_frames_num if ground_truth else pred_frames_num
        assert model_args.pred_frames_num is not None, 'specify pred_frames_num'
        model_args.keypoints_num = dataloader.dataset.keypoints_num
        model = get_model(model_args).to('cuda')
    else:
        msg = "Please provide either a model_name or a load_path to a trained model."
        logger.error(msg)
        raise Exception(msg)

    index = random.randint(0, dataloader.dataset.__len__() - 1) if index is None else index
    data = dataloader.dataset.__getitem__(index)

    for key in ['observed_pose', 'future_pose', 'observed_mask', 'future_mask']:
        if key in data.keys():
            data[key] = data.get(key).unsqueeze(0)

    for key in ['observed_pose', 'observed_mask']:
        if key in data.keys():
            data[key] = data.get(key).to('cuda')

    model.eval()
    with torch.no_grad():
        outputs = model(data)
        assert 'pred_pose' in outputs.keys(), 'outputs of model should include pred_pose'
        data['pred_pose'] = outputs['pred_pose'].detach().cpu()
        if dataloader_args.use_mask:
            assert 'pred_mask' in outputs.keys(), 'outputs of model should include pred_mask'
            data['pred_mask'] = outputs['pred_mask'].detach().cpu()

    for key in ['observed_pose', 'observed_mask']:
        if key in data.keys():
            data[key] = data.get(key).detach().cpu()

    names = []
    poses = []
    masks = []
    images_path = []
    cam_exs = []

    for key in ['observed_pose', 'future_pose', 'pred_pose']:
        if key in data.keys():
            pose = data.get(key).squeeze(0) if dataloader_args.is_interactive else data.get(key)
            poses.append(pose.permute(1, 0, 2))
            names.append(key.split('_')[0])

    for key in ['observed_mask', 'future_mask', 'pred_mask']:
        if key in data.keys():
            mask = data.get(key).squeeze(0) if dataloader_args.is_interactive else data.get(key)
            masks.append(mask.permute(1, 0, 2))

    if 'observed_image' in data.keys():
        images_path.append(data.get('observed_image'))
    if 'future_image' in data.keys():
        images_path.append(data.get('future_image'))
        if 'pred_pose' in data.keys():
            images_path.append(data.get('future_image'))

    if 'observed_cam_ex' in data.keys():
        cam_exs.append(data.get('observed_cam_ex'))
    if 'future_cam_ex' in data.keys():
        cam_exs.append(data.get('future_cam_ex'))
        if 'pred_pose' in data.keys():
            cam_exs.append(data.get('future_cam_ex'))

    cam_in = data.get('cam_in') if 'cam_in' in data.keys() else None

    gif_name = '_'.join((model.args.model_name, dataloader_args.dataset_name.split("/")[-1], str(index)))
    visualizer = Visualizer(dataset_name=dataset_name,
                            images_dir=os.path.join(ROOT_DIR, images_dir if images_dir else ''))
    if dataloader_args.keypoint_dim == 2:
        visualizer.visualizer_2D(names, poses, masks, images_path, gif_name)
    else:
        visualizer.visualizer_3D(names, poses, cam_exs, cam_in, images_path, gif_name)
