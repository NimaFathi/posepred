import logging

import hydra
from omegaconf import DictConfig

from configs.helper import DataloaderArgs, ModelArgs
from data_loader.my_dataloader import get_dataloader
from factory.output_generator import Output_Generator
from models import get_model
from path_definition import HYDRA_PATH
from path_definition import ROOT_DIR
from utils.save_load import load_snapshot, save_args, setup_testing_dir

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="predict")
def generate_output(cfg: DictConfig):
    dataloader_args = DataloaderArgs(cfg.dataloader.dataset_file_name, cfg.keypoint_dim, cfg.interactive,
                                     cfg.persons_num, cfg.use_mask, cfg.skip_num, cfg.dataloader.batch_size,
                                     cfg.dataloader.shuffle, cfg.pin_memory, cfg.num_workers, is_testing=True)
    model_args = ModelArgs(cfg.model.model_name, cfg.use_mask, cfg.keypoint_dim)

    if cfg.train_dataset is not None:
        train_dataloader_args = DataloaderArgs(cfg.train_dataset, cfg.keypoint_dim, cfg.interactive,
                                               cfg.persons_num, cfg.use_mask, cfg.skip_num, 1, False,
                                               cfg.pin_memory, cfg.num_workers)
    else:
        train_dataloader_args = None
    dataloader = get_dataloader(dataloader_args)

    if cfg.load_path:
        model, _, _, _, _ = load_snapshot(cfg.load_path)
    elif model_args.model_name:
        if model_args.model_name == 'nearest_neighbor':
            assert train_dataloader_args is not None, 'Please provide a train_dataset for nearest_neighbor model.'
            train_dataloader = get_dataloader(train_dataloader_args)
            model_args.pred_frames_num = train_dataloader.dataset.future_frames_num
            model_args.keypoints_num = dataloader.dataset.keypoints_num
            model = get_model(model_args).to('cuda')
            model.train_dataloader = train_dataloader
        else:
            model_args.pred_frames_num = cfg.pred_frames_num
            assert model_args.pred_frames_num is not None, 'specify pred_frames_num'
            model_args.keypoints_num = dataloader.dataset.keypoints_num
            model = get_model(model_args).to('cuda')
    else:
        msg = "Please provide either a model_name or a load_path to a trained model."
        logger.error(msg)
        raise Exception(msg)

    save_dir = setup_testing_dir(ROOT_DIR)
    save_args({'dataloader_args': dataloader_args, 'model_args': model.args}, save_dir)

    utput_enerator = Output_Generator(model, dataloader, cfg.interactive, save_dir)
    utput_enerator.generate()


if __name__ == '__main__':
    generate_output()
