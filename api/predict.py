import logging
from logging import config

from args.prediction_args import parse_prediction_args
from data_loader.data_loader import get_dataloader
from factory.predictor import Predictor
from path_definition import ROOT_DIR
from utils.save_load import get_model, load_snapshot, save_args, setup_testing_dir

config.fileConfig('configs/logging.conf')
logger = logging.getLogger('root')

if __name__ == '__main__':

    dataloader_args, model_args, load_path, pred_frames_num, is_interactive, train_dataloader_args = parse_prediction_args()
    dataloader = get_dataloader(dataloader_args)

    if load_path:
        model, _, _, _, _ = load_snapshot(load_path)
    elif model_args.model_name:
        if model_args.model_name == 'nearest_neighbor':
            assert train_dataloader_args is not None, 'Please provide a train_dataset for nearest_neighbor model.'
            train_dataloader = get_dataloader(train_dataloader_args)
            model_args.pred_frames_num = train_dataloader.dataset.future_frames_num
            model_args.keypoints_num = dataloader.dataset.keypoints_num
            model = get_model(model_args)
            model.train_dataloader = train_dataloader
        else:
            model_args.pred_frames_num = pred_frames_num
            assert model_args.pred_frames_num is not None, 'specify pred_frames_num'
            model_args.keypoints_num = dataloader.dataset.keypoints_num
            model = get_model(model_args)
    else:
        msg = "Please provide either a model_name or a load_path to a trained model."
        logger.error(msg)
        raise Exception(msg)

    save_dir = setup_testing_dir(ROOT_DIR)
    save_args({'dataloader_args': dataloader_args, 'model_args': model.args}, save_dir)

    predictor = Predictor(model, dataloader, is_interactive, save_dir)
    predictor.predict()
