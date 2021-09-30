import logging
import os
import random

import hydra
import torch
from omegaconf import DictConfig

from api.helper import DataloaderArgs, ModelArgs
from data_loader.my_dataloader import get_dataloader
from models import get_model
from path_definition import HYDRA_PATH
from path_definition import ROOT_DIR
from utils.save_load import load_snapshot
from utils.lists import dataset
from visualization.visualizer import Visualizer
logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="visualize")
def visualize(cfg: DictConfig):
    assert cfg.dataset in dataset, 'invalid dataset name'
    dataloader_args = DataloaderArgs(cfg.dataloader.dataset_file_name, cfg.keypoint_dim, cfg.interactive,
                                     cfg.persons_num,
                                     cfg.use_mask, cfg.skip_num, cfg.dataloader.batch_size,
                                     cfg.dataloader.shuffle, cfg.pin_memory,
                                     cfg.num_workers, is_testing=not cfg.ground_truth, is_visualizing=True)
    model_args = ModelArgs(cfg.model.model_name, cfg.use_mask, cfg.keypoint_dim)
    dataloader = get_dataloader(dataloader_args)
    if cfg.load_path:
        model, _, _, _, _ = load_snapshot(cfg.load_path)
    elif model_args.model_name:
        model_args.pred_frames_num = dataloader.dataset.future_frames_num if cfg.ground_truth else cfg.pred_frames_num
        assert model_args.pred_frames_num is not None, 'specify pred_frames_num'
        model_args.keypoints_num = dataloader.dataset.keypoints_num
        model = get_model(model_args).to('cuda')
    else:
        msg = "Please provide either a model_name or a load_path to a trained model."
        logger.error(msg)
        raise Exception(msg)
    index = random.randint(0, dataloader.dataset.__len__() - 1) if cfg.index is None else cfg.index
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
    visualizer = Visualizer(dataset_name=cfg.dataset,
                            images_dir=os.path.join(ROOT_DIR, cfg.images_dir if cfg.images_dir else ''))
    if dataloader_args.keypoint_dim == 2:
        visualizer.visualizer_2D(names, poses, masks, images_path, gif_name)
    else:
        visualizer.visualizer_3D(names, poses, cam_exs, cam_in, images_path, gif_name)


if __name__ == '__main__':
    visualize()
