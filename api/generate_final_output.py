import logging
import hydra
from omegaconf import DictConfig

from data_loader.my_dataloader import get_dataloader
from models import MODELS
from factory.output_generator import Output_Generator
from utils.save_load import load_snapshot, setup_testing_dir

from path_definition import HYDRA_PATH
from path_definition import ROOT_DIR

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="predict")
def generate_output(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)

    dataloader = get_dataloader(cfg.test_dataset, cfg.data)

    if cfg.load_path is not None:
        model, _, _, _, _, _, _ = load_snapshot(cfg.load_path)
    else:
        if cfg.model.type == 'nearest_neighbor':
            assert cfg.train_dataset is not None, 'Please provide a train_dataset for nearest_neighbor model.'
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

    save_dir = setup_testing_dir(ROOT_DIR)
    save_args({'dataloader_args': dataloader_args, 'model_args': model.args}, save_dir)

    utput_enerator = Output_Generator(model, dataloader, cfg.interactive, save_dir)
    utput_enerator.generate()


if __name__ == '__main__':
    generate_output()
