from args.evaluation_args import parse_evaluation_args
from data_loader.data_loader import get_dataloader
from utils.save_load import get_model, load_snapshot
from utils.reporter import Reporter
from factory.evaluator import Evaluator

if __name__ == '__main__':

    dataloader_args, model_args, load_path, is_interactive, distance_loss, rounds_num, train_dataloader_args = parse_evaluation_args()
    dataloader = get_dataloader(dataloader_args)
    reporter = Reporter()

    if load_path:
        model, _, _, _, _ = load_snapshot(load_path)
    elif model_args.model_name:
        model_args.pred_frames_num = dataloader.dataset.future_frames_num
        model_args.keypoints_num = dataloader.dataset.keypoints_num
        model = get_model(model_args)
        if model_args.model_name == 'nearest_neighbor':
            assert train_dataloader_args is not None, 'Please provide a train_dataset for nearest_neighbor model.'
            model.train_dataloader = get_dataloader(train_dataloader_args)
    else:
        raise Exception("Please provide either a model_name or a load_path to a trained model.")

    evaluator = Evaluator(model, dataloader, reporter, is_interactive, distance_loss, rounds_num)
    evaluator.evaluate()
