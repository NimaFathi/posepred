import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
import random


class Basic_Dataset(Dataset):
    def __init__(self, dataset_path, use_mask, skip_frame, is_multi_person):
        data = pd.read_csv(dataset_path)
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
        self.use_mask = use_mask
        self.skip_frame = skip_frame
        self.is_multi_person = is_multi_person

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
            obs_pose = torch.tensor(
                [person_obs_pose_frames[i] for i in range(0, self.obs_frames_num, self.skip_frame + 1)])
            obs_vel = (obs_pose[1:] - obs_pose[:-1])

            person_future_pose_frames = seq.future_pose[person_index]
            future_pose = torch.tensor(
                [person_future_pose_frames[i] for i in range(0, self.future_frames_num, self.skip_frame + 1)])
            future_vel = torch.cat((future_pose[0] - obs_pose[-1]).unsqueeze(0), future_pose[1:] - future_pose[:-1])

            outputs_p.append(obs_pose)
            outputs_p.append(obs_vel)
            outputs_p.append(future_pose)
            outputs_p.append(future_vel)

            if self.use_mask:
                person_obs_mask_frames = seq.observed_mask[person_index]
                obs_mask = torch.tensor(
                    [person_obs_mask_frames[i] for i in range(0, self.obs_frames_num, self.skip_frame + 1)])
                person_future_mask_frames = seq.future_mask[person_index]
                true_mask = torch.tensor(
                    [person_future_mask_frames[i] for i in range(0, self.future_frames_num, self.skip_frame + 1)])
                outputs.append(obs_mask)
                outputs.append(true_mask)
            outputs.append(tuple(outputs_p))

        if self.is_multi_person:
            return outputs
        else:
            return random.choice(outputs)
