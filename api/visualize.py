import os
import logging
import hydra
from omegaconf import DictConfig
import random
import torch

from data_loader.my_dataloader import get_dataloader
from data_loader import DATASETS
from models import MODELS
from utils.save_load import load_snapshot
from visualization.visualizer import Visualizer

from path_definition import HYDRA_PATH
from path_definition import ROOT_DIR

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="visualize")
def visualize(cfg: DictConfig):
    assert cfg.dataset_name in DATASETS, 'dataset_name chioces: ' + str(DATASETS)
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)

    # prepare data
    dataloader = get_dataloader(cfg.dataset, cfg.data)
    index = random.randint(0, dataloader.dataset.__len__() - 1) if cfg.index is None else cfg.index
    data = dataloader.dataset.__getitem__(index)
    for key in ['observed_pose', 'future_pose', 'observed_mask', 'future_mask']:
        if key in data.keys():
            data[key] = data.get(key).unsqueeze(0)
    for key in ['observed_pose', 'observed_mask']:
        if key in data.keys():
            data[key] = data.get(key).to(cfg.device)

    # prepare model
    if cfg.load_path is not None:
        model, _, _, _, _, _, _ = load_snapshot(cfg.load_path)
    else:
        cfg.model.keypoint_dim = cfg.data.keypoint_dim
        cfg.model.keypoints_num = dataloader.dataset.keypoints_num
        cfg.model.use_mask = cfg.data.use_mask
        cfg.model.pred_frames_num = dataloader.dataset.future_frames_num if cfg.ground_truth else cfg.pred_frames_num
        assert cfg.model.pred_frames_num is not None, 'specify pred_frames_num'
        model = MODELS[cfg.model.type](cfg.model)
        if cfg.model.type == 'nearest_neighbor':
            model.train_dataloader = get_dataloader(cfg.model.train_dataset, cfg.data)

    # predict
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        assert 'pred_pose' in outputs.keys(), 'outputs of model should include pred_pose'
        data['pred_pose'] = outputs['pred_pose']
        if cfg.data.use_mask:
            assert 'pred_mask' in outputs.keys(), 'outputs of model should include pred_mask'
            data['pred_mask'] = outputs['pred_mask']

    for key in ['observed_pose', 'observed_mask', 'pred_pose', 'pred_mask']:
        if key in data.keys():
            if data[key].is_cuda:
                data[key] = data[key].detach().cpu()

    names = []
    poses = []
    masks = []
    images_path = []
    cam_exs = []

    for key in ['observed_pose', 'future_pose', 'pred_pose']:
        if key in data.keys():
            pose = data.get(key).squeeze(0) if cfg.data.is_interactive else data.get(key)
            poses.append(pose.permute(1, 0, 2))
            names.append(key.split('_')[0])

    for key in ['observed_mask', 'future_mask', 'pred_mask']:
        if key in data.keys():
            mask = data.get(key).squeeze(0) if cfg.data.is_interactive else data.get(key)
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

    gif_name = '_'.join((cfg.model.type, cfg.dataset.split("/")[-1], str(index)))
    visualizer = Visualizer(dataset_name=cfg.dataset_name,
                            images_dir=os.path.join(ROOT_DIR, cfg.images_dir if cfg.images_dir else ''))
    if cfg.data.keypoint_dim == 2:
        visualizer.visualizer_2D(names, poses, masks, images_path, gif_name)
    else:
        visualizer.visualizer_3D(names, poses, cam_exs, cam_in, images_path, gif_name)


if __name__ == '__main__':
    visualize()
