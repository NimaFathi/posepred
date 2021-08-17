import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval


class BasicDataset(Dataset):
    def __init__(self, dataset_path, data_dim, is_testing, use_mask, skip_frame):
        data = pd.read_csv(dataset_path)
        for col in list(data.columns[1:].values):
            try:
                data.loc[:, col] = data.loc[:, col].apply(lambda x: literal_eval(x))
            except:
                raise Exception("data must be convertable to valid data-scructures")

        self.data = data.copy().reset_index(drop=True)
        self.dim = data_dim
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.skip_frame = skip_frame

        seq = self.data.iloc[0]
        self.joints = len(seq.observed_pose[0][0]) / self.dim
        self.obs_frames_num = len(seq.observed_pose[0])
        if not self.is_testing:
            self.future_frames_num = len(seq.future_pose[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data.iloc[index]

        obs_pose = self.get_tensor(seq, 'observed_pose', self.obs_frames_num)
        obs_vel = (obs_pose[:, 1:, :] - obs_pose[:, :-1, :])
        outputs = [obs_pose, obs_vel]

        if self.use_mask:
            obs_mask = self.get_tensor(seq, 'observed_mask', self.obs_frames_num)
            outputs.append(obs_mask)

        if not self.is_testing:
            assert len(seq.observed_pose) == len(seq.future_pose), "unequal persons in observed and future frames."
            future_pose = self.get_tensor(seq, 'future_pose', self.future_frames_num)
            future_vel = torch.cat(((future_pose[:, 0, :] - obs_pose[:, -1, :]).unsqueeze(1),
                                    future_pose[:, 1:, :] - future_pose[:, :-1, :]), 1)
            outputs += [future_pose, future_vel]

            if self.use_mask:
                future_mask = self.get_tensor(seq, 'future_mask', self.future_frames_num)
                outputs.append(future_mask)

        return tuple(outputs)

    def get_tensor(self, seq, segment, frames_num):
        assert segment in seq, 'No segment named: ' + segment
        persons_num = len(seq.observed_pose)
        return torch.tensor(
            [[seq[segment][person_idx][frame_idx]
              for frame_idx in range(0, frames_num, self.skip_frame + 1)]
             for person_idx in range(persons_num)])
