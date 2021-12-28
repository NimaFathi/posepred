import os
import logging
import hydra
from omegaconf import DictConfig
import json
import torch

from data_loader import get_dataloader
from models import MODELS
from factory.output_generator import Output_Generator
from utils.save_load import load_snapshot, setup_testing_dir

from path_definition import HYDRA_PATH

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="generate_output")
def generate_output(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)

    dataloader = get_dataloader(cfg.dataset, cfg.data)

    if cfg.load_path is not None:
        model, _, _, _, _, _, _ = load_snapshot(cfg.load_path)
    else:
        cfg.model.keypoint_dim = cfg.data.keypoint_dim
        cfg.model.pred_frames_num = cfg.pred_frames_num
        cfg.model.keypoints_num = dataloader.dataset.keypoints_num
        cfg.model.use_mask = cfg.data.use_mask
        model = MODELS[cfg.model.type](cfg.model)
        if cfg.model.type == 'nearest_neighbor':
            model.train_dataloader = get_dataloader(cfg.model.train_dataset, cfg.data)
    cfg.save_dir = os.getcwd()
    setup_testing_dir(cfg.save_dir)

    # 3DPW
    # with open("/home/rmool/posepred/preprocessed_data/3dpw_test_in.json", "r") as read_file:
    #     data = json.load(read_file)
    # out_data = []
    # for i in range(len(data)):
    #     lp = []
    #     for j in range(len(data[i])):
    #         pose = torch.tensor(data[i][j]).unsqueeze(0).to(cfg.device)
    #         inputs = {'observed_pose': pose}
    #         outputs = model(inputs)
    #         pred = outputs['pred_pose'].squeeze(0)
    #         lp.append(pred.tolist())
    #     out_data.append(lp)
    # with open("/home/rmool/posepred/preprocessed_data/3dpw_test_out.json", 'w') as f:
    #     json.dump(out_data, f)

    with open("/home/rmool/posepred/preprocessed_data/posetrack_test_in.json", "r") as read_file:
        data = json.load(read_file)
    with open("/home/rmool/posepred/preprocessed_data/posetrack_test_masks_in.json", "r") as read_file:
        data_m = json.load(read_file)
    out_data = []
    out_mask = []
    for i in range(len(data)):
        lp = []
        lm = []
        for j in range(len(data[i])):
            pose = torch.tensor(data[i][j]).unsqueeze(0).to(cfg.device)
            mask = torch.tensor(data_m[i][j]).unsqueeze(0).to(cfg.device)
            inputs = {'observed_pose': pose}
            outputs = model(inputs)
            pred = outputs['pred_pose'].squeeze(0)
            m = mask[:, -1:, :]
            mask_preds = torch.cat((m, m, m, m, m, m, m, m, m, m, m, m, m, m), 1).squeeze(0)
            lp.append(pred.tolist())
            lm.append(mask_preds.detach().cpu().numpy().round().tolist())
        out_data.append(lp)
        out_mask.append(lm)
    with open("/home/rmool/posepred/preprocessed_data/posetrack_test_out.json", 'w') as f:
        json.dump(out_data, f)
    with open("/home/rmool/posepred/preprocessed_data/posetrack_test_out_masks.json", 'w') as f:
        json.dump(out_mask, f)

    # output_enerator = Output_Generator(model, dataloader, cfg.save_dir, cfg.device)
    # output_enerator.generate()


if __name__ == '__main__':
    generate_output()
