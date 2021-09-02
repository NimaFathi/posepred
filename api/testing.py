from path_definition import ROOT_DIR
from args.testing_args import parse_testing_args
from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_snapshot, save_args, setup_testing_dir
from factory.tester import Tester

if __name__ == '__main__':

    dataloader_args, model_args, load_path, pred_frames_num, is_interactive = parse_testing_args()
    dataloader = get_dataloader(dataloader_args)

    if load_path:
        model, _, _, _, _ = load_snapshot(load_path)
    elif model_args.model_name:
        model_args.pred_frames_num = pred_frames_num
        assert model_args.pred_frames_num is not None, 'specify pred_frames_num'
        model_args.keypoints_num = dataloader.dataset.keypoints_num
        model = get_model(model_args)
    else:
        raise Exception("Please provide either a model_name or a load_path to a trained model.")

    save_dir = setup_testing_dir(ROOT_DIR)
    save_args({'dataloader_args': dataloader_args, 'model_args': model.args}, save_dir)

    tester = Tester(model, dataloader, is_interactive, save_dir)
    tester.test()
