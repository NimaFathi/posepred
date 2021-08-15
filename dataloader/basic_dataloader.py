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
        seq = self.data.iloc[0]
        self.obs_frames_num = len(seq.observed_pose[0])
        self.future_frames_num = len(seq.future_pose[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data.iloc[index]
        assert len(seq.observed_pose) == len(
            seq.future_pose), "number of persons must be equal in observed and future frames."

        seq_persons_num = len(seq.observed_pose)

        outputs = []
        for person_index in range(seq_persons_num):
            outputs_p = []

            person_obs_pose_frames = seq.observed_pose[person_index]
            obs_pose = torch.tensor([person_obs_pose_frames[i] for i in range(0, self.obs_frames_num, self.args.skip + 1)])
            obs_vel = (obs_pose[1:] - obs_pose[:-1])

            person_future_pose_frames = seq.future_pose[person_index]
            future_pose = torch.tensor(
                [person_future_pose_frames[i] for i in range(0, self.future_frames_num, self.args.skip + 1)])
            future_vel = torch.cat((future_pose[0] - obs_pose[-1]).unsqueeze(0), future_pose[1:] - future_pose[:-1])

            outputs_p.append(obs_pose)
            outputs_p.append(obs_vel)
            outputs_p.append(future_pose)
            outputs_p.append(future_vel)

            if self.args.use_mask:
                person_obs_mask_frames = seq.observed_mask[person_index]
                obs_mask = torch.tensor([person_obs_mask_frames[i] for i in range(0, self.obs_frames_num, self.args.skip + 1)])

                person_future_mask_frames = seq.future_mask[person_index]
                true_mask = torch.tensor([person_future_mask_frames[i] for i in range(0, self.future_frames_num, self.args.skip + 1)])

                outputs.append(obs_mask)
                outputs.append(true_mask)

            outputs.append(outputs_p)
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
