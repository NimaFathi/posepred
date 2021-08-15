import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ast import literal_eval


class Dataloader(Dataset):
    def __init__(self, args):
        self.args = args
        data = pd.read_csv('../preprocessed_data/simple_dataset.csv')
        for col in list(data.columns.values):
            try:
                data.loc[:, col] = data.loc[:, col].apply(lambda x: literal_eval(x))
            except Exception as e:
                print("Exception of type", type(e), "occurred: ", e.args)
                continue
        self.data = data.copy().reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data.iloc[index]
        assert len(seq.observed_pose) == len(seq.future_pose), "number of persons must be equal in observed and future frames."

        outputs = []
        for index_p, obs_seq_person in seq.observed_pose:
            outputs_p = []

            obs_pose = torch.tensor([obs_seq_person[i] for i in range(0, self.args.input, self.args.skip)])
            obs_vel = (obs_pose[1:] - obs_pose[:-1])

            future_seq_person = seq.future_pose[index_p]
            future_pose = torch.tensor([future_seq_person for i in range(0, self.args.output, self.args.skip)])
            future_vel = torch.cat((future_pose[0] - obs_pose[-1]).unsqueeze(0), future_pose[1:] - future_pose[:-1])

            outputs_p.append(obs_pose)
            outputs_p.append(obs_vel)
            outputs_p.append(future_pose)
            outputs_p.append(future_vel)
            outputs.append(outputs_p)

        if self.fname == "posetrack_":
            obs_mask = torch.tensor([seq.Mask[i] for i in range(0, self.args.output, self.args.skip)])
            true_mask = torch.tensor([seq.Future_Mask[i] for i in range(0, self.args.output, self.args.skip)])
            outputs.append(obs_mask)
            outputs.append(true_mask)
        return tuple(outputs)


# def data_loader_lstm_vel(args, data, fname):
#     dataset = Dataloader(args, data, fname)
#     dataloader = DataLoader(
#         dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
#         pin_memory=args.pin_memory)
#     return dataloader


# data = pd.read_csv('../preprocessed_data/simple_dataset.csv')
# for col in list(data.columns.values):
#     try:
#         data.loc[:, col] = data.loc[:, col].apply(lambda x: literal_eval(x))
#     except Exception as e:
#         print("Exception of type", type(e), "occurred: ", e.args)
#         continue
# self_data = data.copy().reset_index(drop=True)
#
# seq = self_data.iloc[0]
# print(seq.observed_pose)

