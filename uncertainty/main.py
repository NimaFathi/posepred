import os
from argparse import Namespace

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from .model.dc.deep_clustering import DCModel
from .model.dc.train_dc_seq import train_dc_model, cluster
from .model.lstm.lstm import LstmAutoEncoder, EncoderWrapper
from .model.lstm.train_lstm import train_lstm_model
from .utils.train_utils import save_model, save_model_results_dict
from .utils.uncertainty import *
from .utils.dataset_utils import TRAIN_K, VALID_K, TEST_K, INCLUDED_JOINTS_COUNT, SKIP_RATE, SCALE_RATIO, H36_ACTIONS, \
    DIM
from .utils.prediction_util import get_prediction_model_dict, get_dataloader_dict, PRED_MODELS, PRED_MODELS_ARGS


# def load_dataset(dataset_path: str, dataset_name: str, input_n: int, output_n: int, test: bool, mode: str, vel: bool) -> dict:
#     dataset = {}
#     skip_rate, scale = SKIP_RATE[dataset_name], SCALE_RATIO[dataset_name]
#     if dataset_name == 'Human36m':
#         train_inp = 0
#         if mode == 'rejection':
#             train_inp = output_n
#         if not test:
#             print("TRAIN SET")
#             dataset[TRAIN_K] = Human36M(
#                 dataset_path, train_inp, output_n, skip_rate, scale, H36_ACTIONS, split=0, apply_joints_to_include=True, vel=vel)
#             dataset[VALID_K] = Human36M(
#                 dataset_path, train_inp, output_n, skip_rate, scale, H36_ACTIONS, split=1, apply_joints_to_include=True, vel=vel)
#         dataset[TEST_K] = Human36M(
#             dataset_path, output_n, output_n, skip_rate, 1, H36_ACTIONS, split=2, vel=vel)  # scale must be 1
#     if dataset_name == 'AMASS':
#         if not test:
#             dataset[TRAIN_K] = Amass(dataset_path, 0, output_n, skip_rate, split=0, apply_joints_to_include=True)
#             dataset[VALID_K] = Amass(dataset_path, 0, output_n, skip_rate, split=1, apply_joints_to_include=True)
#         dataset[TEST_K] = Amass(dataset_path, input_n, output_n, skip_rate, split=2, apply_joints_to_include=False)
#     if dataset_name == '3DPW':
#         if not test:
#             dataset[TRAIN_K] = Dpw3(dataset_path, 0, output_n, skip_rate, split=0, apply_joints_to_include=True)
#             dataset[VALID_K] = Dpw3(dataset_path, 0, output_n, skip_rate, split=1, apply_joints_to_include=True)
#         dataset[TEST_K] = Dpw3(dataset_path, input_n, output_n, skip_rate, split=2, apply_joints_to_include=False)
#     return dataset


def load_dc_model(dataset_name: str, n_clusters: int, dc_model_path: str, dev='cuda'):
    lstm_ae = LstmAutoEncoder(pose_dim=INCLUDED_JOINTS_COUNT[dataset_name]).to(dev)
    dc_model = DCModel(lstm_ae, n_clusters=n_clusters).to(dev)
    dc_model.load_state_dict(torch.load(dc_model_path))
    return dc_model


def init_dc_train(dataset_args: Namespace, model_args: Namespace, data_loader: dict, train_ds: Dataset,
        output_path: str, vel: bool, dev='cuda'):
    lstm_ae = LstmAutoEncoder(pose_dim=INCLUDED_JOINTS_COUNT[dataset_args.dataset], dev=dev)
    if not model_args.lstm_path is None:
        lstm_ae = torch.load(model_args.lstm_path)
    else:
        train_lstm_model(model_args, lstm_ae, data_loader[TRAIN_K], data_loader[VALID_K], dev=dev)
        torch.save(lstm_ae, 'lstm_model_hdim8.pt')
    lstm_ae.eval()
    lstm_ae.to(dev)
    encoder = EncoderWrapper(lstm_ae).to(dev)
    initial_clusters = cluster(train_ds, encoder, model_args.n_clusters, device)
    dc_model = DCModel(lstm_ae=lstm_ae, n_clusters=model_args.n_clusters,
                       initial_clusters=initial_clusters,
                       device=dev)
    best_model = train_dc_model(model_args, dc_model, train_ds, dataset_args.batch_size, num_workers=dataset_args.num_workers,
                   dev=dev)
    save_model(dc_model, output_path)
    save_model(best_model, output_path, best=True)
    return dc_model


# def evaluate_uncertainty_and_loss(dataset_args: Namespace, evaluation_args: Namespace, dc_model: DCModel,
#         loader: DataLoader, ds: str, mode: str, vel: bool, additional_loader=None, draw=False, dev='cuda') -> dict:
#     """
#     Evaluates uncertainty and loss for a given prediction model or dictionary of the results.
#     :param dataset_args:
#     :param evaluation_args:
#     :param dc_model:
#     :param test_loader:
#     :param pred_model:
#     :param dev:
#     :return:
#     """
#     pred_model = evaluation_args.pred_model
#     dataset_name, b_size = dataset_args.dataset, dataset_args.batch_size
#     input_n, output_n = dataset_args.input_n, dataset_args.output_n
#     is_dict = evaluation_args.model_dict_path is not None
#     maha = torch.load('maha.pt')
#     std = None
# #    _, std = get_dc_model_attrs(dc_model, additional_loader)
# #    torch.save(std, 'std_0.3.pt')
#     std = torch.load('std_0.3.pt')
#     if is_dict:
#         model_dict = torch.load(evaluation_args.model_dict_path)
#     elif not pred_model is None:
#         if pred_model == 'zerovel':
#             model = PRED_MODELS[pred_model](output_n)
#         else:
#             assert evaluation_args.model_path is not None and pred_model is not None
#             model = PRED_MODELS[pred_model](**PRED_MODELS_ARGS[pred_model])
#             model.load_state_dict(torch.load(evaluation_args.model_path))
#         model.eval()
#         model.to(dev)
#         model_dict = get_prediction_model_dict(model, loader, input_n, output_n, dataset_name, vel, dev, evaluation_args.dropout)
#     if mode == 'model':
#         if not draw:
#             evaluation_dict = calculate_dict_uncertainty_and_mpjpe(dataset_name, model_dict, dc_model, b_size, maha, dev)
#     else:
#         loader_vel = vel if ds == TEST_K else False
#         loader_dict = get_dataloader_dict(additional_loader, input_n, output_n, dataset_name, loader_vel, dev)
# #        _, std = get_dc_model_attrs(dc_model, additional_loader)
# #        torch.save(loader_dict, 'TRAIN_dict.pt')
#         evaluation_dict = calculate_dataloader_dict_uncertainty(dataset_name, loader_dict, dc_model, b_size, std, ds, dev)
#     return evaluation_dict


# def save_results_to_file(results: dict, output_path: str, prediction_model_path: str, mode: str):
#     """
#     Beautifies, prints and saves the experiments' results based on the corresponding mode
#     :param results:
#     :param output_path:
#     :param prediction_model_path:
#     """
#     prediction_model_name = prediction_model_path.split('/')[-1] if not prediction_model_path is None else "rej"
#     output_file = open(f'{output_path}/{prediction_model_name}_uncertainty_{mode}_results.txt', 'w')
#     lines = []
#     if mode == 'rejection':
#         for k in results.keys():
#             for f in results[k].keys():
#                 line = f'** {k}-{f}: {results[k][f]} **\n'
#                 lines.append(line)
#     elif mode == 'extreme':
#         for k in results.keys():
#             for f in results[k].keys():
#                 if f != INP_K and f != OUT_K:
#                     line = f'** {k}-{f}: {results[k][f]} **\n'
#                     lines.append(line)
#     else:
#         if "loss" not in results:
#             lines.append(f'** Uncertainty:{results["uncertainty"]}\n')
#         else:
#             lines.append(f'** Uncertainty:{results["uncertainty"]}, MPJPE:{results["loss"]}\n')
#     for l in lines:
#         print(l, end='')
#     output_file.writelines(lines)
#     output_file.close()
#
#
# def init_data_loaders(dataset_args: Namespace, test: bool, ds: str, mode: str, vel: bool):
#     """
#     Initialize training and validation DataLoader instances based on given dataset name.
#     :param dataset_args: Human36m, AMASS or 3DPW
#     :param test: Indicates whether train dataloader is required
#     """
#     test_only = (test and ds == TEST_K)
#     dataset = load_dataset(dataset_args.dataset_path, dataset_args.dataset,
#                            dataset_args.input_n, dataset_args.output_n, test_only, mode, vel)
#     data_loader = {
#         TEST_K: DataLoader(dataset[TEST_K], batch_size=dataset_args.batch_size, shuffle=False, pin_memory=True),
#     }
#     if not test_only:
#         data_loader[TRAIN_K] = DataLoader(dataset[TRAIN_K], batch_size=dataset_args.batch_size, shuffle=True,
#                                           pin_memory=True)
#         data_loader[VALID_K] = DataLoader(dataset[VALID_K], batch_size=dataset_args.batch_size, shuffle=False,
#                                           pin_memory=True)
#         return data_loader, dataset[TRAIN_K]
#     return data_loader, None
#
#
# def main(main_args: Namespace, dataset_args: Namespace, model_args: Namespace, evaluation_args: Namespace):
#     """
#     Main function. Loads the dataset, trains the uncertainty evaluation pipeline and returns the result of the
#     provided pipeline.
#     :param main_args:
#     :param dataset_args:
#     :param model_args:
#     :param evaluation_args:
#     """
#     dev, dataset_name, test, mode, vel, ds = main_args.device, dataset_args.dataset, main_args.test, main_args.mode, main_args.vel, main_args.additional_ds
#     data_loader, train_ds = init_data_loaders(dataset_args, test, ds, mode, vel)
#     output_path = evaluation_args.output_path
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     if test:
#         dc_model = load_dc_model(dataset_name, model_args.n_clusters, evaluation_args.dc_model_path, vel=vel)
#         loader = data_loader[TEST_K]
#     else:
#         dc_model = init_dc_train(dataset_args, model_args, data_loader, train_ds, output_path, vel, dev)
#         loaderr = data_loader[TRAIN_K]
#     dc_model.eval()
#     dc_model.to(dev)
#     if mode == 'model':
#         additional_loader = data_loader[ds]
#     elif mode in ('auroc', 'extreme', 'rejection', 'ds'):
#         additional_loader = data_loader[ds]
#     results = evaluate_uncertainty_and_loss(dataset_args, evaluation_args, dc_model, loader, ds, mode, vel,                                            dev=main_args.device, additional_loader=additional_loader, draw=main_args.plot)
#     save_results_to_file(results, output_path, evaluation_args.model_path, mode=mode)