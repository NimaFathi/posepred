import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from numpy import random


class InteractiveDataset(Dataset):
    def __init__(self, dataset_path, keypoint_dim, persons_num, is_testing, use_mask, skip_frame, is_visualizing):
        data = pd.read_csv(dataset_path)
        for col in list(data.columns[1:].values):
            try:
                data.loc[:, col] = data.loc[:, col].apply(lambda x: literal_eval(x))
            except:
                raise Exception("Each row must be convertable to python list")

        self.data = data.copy().reset_index(drop=True)
        self.persons_num = persons_num
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.skip_frame = skip_frame
        self.is_visualizing = is_visualizing

        seq = self.data.iloc[0]
        self.keypoint_dim = keypoint_dim
        self.keypoints_num = len(seq.observed_pose[0][0]) / self.keypoint_dim
        self.obs_frames_num = len(seq.observed_pose[0])
        if not self.is_testing:
            self.future_frames_num = len(seq.future_pose[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data.iloc[index]
        persons_in_seq = self.select_persons(seq)

        try:
            obs_pose = self.get_tensor(seq, 'observed_pose', persons_in_seq, self.obs_frames_num)
            obs_vel = (obs_pose[:, 1:, :] - obs_pose[:, :-1, :])
            outputs = [obs_pose, obs_vel]
        except:
            print('faulty row skipped.')
            return self.__getitem__((index + 1) % self.__len__())

        if self.use_mask:
            obs_mask = self.get_tensor(seq, 'observed_mask', persons_in_seq, self.obs_frames_num)
            outputs.append(obs_mask)

        if not self.is_testing:
            assert len(seq.observed_pose) == len(seq.future_pose), "unequal persons in observed and future frames."
            future_pose = self.get_tensor(seq, 'future_pose', persons_in_seq, self.future_frames_num)
            future_vel = torch.cat(((future_pose[:, 0, :] - obs_pose[:, -1, :]).unsqueeze(1),
                                    future_pose[:, 1:, :] - future_pose[:, :-1, :]), 1)
            outputs += [future_pose, future_vel]

            if self.use_mask:
                future_mask = self.get_tensor(seq, 'future_mask', persons_in_seq, self.future_frames_num)
                outputs.append(future_mask)

        if self.is_visualizing:

            # print(seq['observed_image_path'])
            # print('observed_image_path' in seq.keys())

        return tuple(outputs)

    def select_persons(self, seq):
        persons_in_seq = list(range(len(seq['observed_pose'])))
        if self.persons_num < len(persons_in_seq):
            return random.choice(persons_in_seq, self.persons_num, replace=False)
        else:
            return persons_in_seq

    def get_tensor(self, seq, segment, persons_in_seq, frames_num):
        assert segment in seq, 'No segment named: ' + segment
        result = torch.tensor(
            [[seq[segment][person_idx][frame_idx]
              for frame_idx in range(0, frames_num, self.skip_frame + 1)]
             for person_idx in persons_in_seq], dtype=torch.float32)

        if len(persons_in_seq) < self.persons_num:
            padding = torch.zeros(self.persons_num - len(persons_in_seq), result.shape[1], result.shape[2])
            result = torch.cat((result, padding), dim=0)

        return result
