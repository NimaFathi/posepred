import os
import logging
import hydra
from omegaconf import DictConfig
import random
import torch

from data_loader import get_dataloader, DATASETS, VISUALIZING_TYPES
from models import MODELS
from utils.save_load import load_snapshot
from utils.others import dict_to_device
from visualization.visualizer import Visualizer

from path_definition import HYDRA_PATH
from path_definition import ROOT_DIR

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="visualize")
def visualize(cfg: DictConfig):
    assert cfg.dataset_type in DATASETS, 'dataset_type chioces: ' + str(DATASETS)
    showing = cfg.showing.strip().split('-')
    for k in showing:
        if k not in VISUALIZING_TYPES:
            raise Exception(
                'options for showing are: ' + str(VISUALIZING_TYPES) + '''\nuse '-' to seperate different types.''')
    if 'future' in showing and cfg.data.is_testing:
        raise Exception('do not have access to future frames when data.is_testing is true.')

    # prepare data
    dataloader = get_dataloader(cfg.dataset, cfg.data)
    index = random.randint(0, dataloader.dataset.__len__() - 1) if cfg.index is None else cfg.index
    data = dataloader.dataset.__getitem__(index)
    for key in ['observed_pose', 'future_pose', 'observed_mask', 'future_mask', 'observed_noise']:
        if key in data.keys():
            data[key] = data[key].unsqueeze(0)

    # prepare model
    if cfg.load_path is not None:
        model, _, _, _, _, _, _ = load_snapshot(cfg.load_path)
    else:
        cfg.model.keypoint_dim = cfg.data.keypoint_dim
        cfg.model.keypoints_num = dataloader.dataset.keypoints_num
        cfg.model.use_mask = cfg.data.use_mask
        cfg.model.pred_frames_num = dataloader.dataset.future_frames_num if cfg.pred_frames_num is None else cfg.pred_frames_num
        assert cfg.model.pred_frames_num is not None, 'specify pred_frames_num or set data.is_testing=false'
        model = MODELS[cfg.model.type](cfg.model)
        if cfg.model.type == 'nearest_neighbor':
            model.train_dataloader = get_dataloader(cfg.model.train_dataset, cfg.data)

    # predict
    model = model.to(cfg.device).eval()
    with torch.no_grad():
        outputs = model(dict_to_device(data, cfg.device))
        assert 'pred_pose' in outputs.keys(), 'outputs of model should include pred_pose'
        data['pred_pose'] = outputs['pred_pose']
        if cfg.data.use_mask:
            assert 'pred_mask' in outputs.keys(), 'outputs of model should include pred_mask'
            data['pred_mask'] = outputs['pred_mask']
        if 'completed' in showing:
            assert 'comp_pose' in outputs.keys(), 'outputs of model should include comp_pose'
            data['comp_pose'] = outputs['comp_pose']

    names = []
    poses = []
    masks = []
    images_path = []
    cam_exs = []

    if 'observed' in showing:
        names.append('observed')
        pose = data['observed_pose'].squeeze(0) if cfg.data.is_interactive else data['observed_pose']
        poses.append(pose.permute(1, 0, 2))
        if 'observed_mask' in data.keys():
            mask = data['observed_mask'].squeeze(0) if cfg.data.is_interactive else data['observed_mask']
            masks.append(mask.permute(1, 0, 2))
        else:
            masks.append(None)
        image_path = data['observed_image'] if 'observed_image' in data.keys() else None
        images_path.append(image_path)
        cam_ex = data['observed_cam_ex'] if 'observed_cam_ex' in data.keys() else None
        cam_exs.append(cam_ex)

    if 'completed' in showing:
        names.append('completed')
        pose = data['comp_pose'].squeeze(0) if cfg.data.is_interactive else data['comp_pose']
        poses.append(pose.permute(1, 0, 2))
        masks.append(None)
        image_path = data['observed_image'] if 'observed_image' in data.keys() else None
        images_path.append(image_path)
        cam_ex = data['observed_cam_ex'] if 'observed_cam_ex' in data.keys() else None
        cam_exs.append(cam_ex)

    if 'future' in showing:
        names.append('future')
        pose = data['future_pose'].squeeze(0) if cfg.data.is_interactive else data['future_pose']
        poses.append(pose.permute(1, 0, 2))
        if 'future_mask' in data.keys():
            mask = data['future_mask'].squeeze(0) if cfg.data.is_interactive else data['future_mask']
            masks.append(mask.permute(1, 0, 2))
        else:
            masks.append(None)
        image_path = data['future_image'] if 'future_image' in data.keys() else None
        images_path.append(image_path)
        cam_ex = data['future_cam_ex'] if 'future_cam_ex' in data.keys() else None
        cam_exs.append(cam_ex)

    if 'predicted' in showing:
        names.append('predicted')
        pose = data['pred_pose'].squeeze(0) if cfg.data.is_interactive else data['pred_pose']
        poses.append(pose.permute(1, 0, 2))
        if 'pred_mask' in data.keys():
            mask = data['pred_mask'].squeeze(0) if cfg.data.is_interactive else data['pred_mask']
            masks.append(mask.permute(1, 0, 2))
        else:
            masks.append(None)
        image_path = data['future_image'] if 'future_image' in data.keys() else None
        images_path.append(image_path)
        cam_ex = data['future_cam_ex'] if 'future_cam_ex' in data.keys() else None
        cam_exs.append(cam_ex)

    cam_in = data.get('cam_in') if 'cam_in' in data.keys() else None

    for i, p in enumerate(poses):
        if p is not None and p.is_cuda:
            poses[i] = p.detach().cpu()

    for i, m in enumerate(masks):
        if m is not None and m.is_cuda:
            masks[i] = m.detach().cpu()

    visualizer = Visualizer(dataset_name=cfg.dataset_type, parent_dir=os.getcwd(), images_dir=cfg.images_dir)
    gif_name = '_'.join((cfg.model.type, cfg.dataset.split("/")[-1], str(index)))
    if cfg.data.keypoint_dim == 2:
        visualizer.visualizer_2D(names, poses, masks, images_path, gif_name)
    else:
        visualizer.visualizer_3D(names, poses, cam_exs, cam_in, images_path, gif_name)


if __name__ == '__main__':
    visualize()
