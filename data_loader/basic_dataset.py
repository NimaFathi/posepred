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
        persons_num = len(seq.observed_pose)

        assert 'observed_pose' in seq, 'there is no observed_pose in dataset.'
        obs_pose = torch.tensor(
            [[seq['observed_pose'][person_idx][frame_idx]
              for frame_idx in range(0, self.obs_frames_num, self.skip_frame + 1)]
             for person_idx in range(persons_num)])
        obs_vel = (obs_pose[:, 1:, :] - obs_pose[:, :-1, :])
        outputs = [obs_pose, obs_vel]

        if self.use_mask:
            assert 'observed_mask' in seq, 'use_mask is true but there is no observed_mask in dataset.'
            obs_mask = torch.tensor(
                [[seq.observed_mask[person_idx][frame_idx]
                  for frame_idx in range(0, self.obs_frames_num, self.skip_frame + 1)]
                 for person_idx in range(persons_num)])
            outputs.append(obs_mask)

        if not self.is_testing:
            assert 'future_pose' in seq, 'is_testing is false but there is no future_pose in dataset.'
            assert len(seq.observed_pose) == len(seq.future_pose), "unequal persons in observed and future frames."
            future_pose = torch.tensor(
                [[seq.future_pose[person_idx][frame_idx]
                  for frame_idx in range(0, self.future_frames_num, self.skip_frame + 1)]
                 for person_idx in range(persons_num)])
            future_vel = torch.cat(((future_pose[:, 0, :] - obs_pose[:, -1, :]).unsqueeze(1),
                                    future_pose[:, 1:, :] - future_pose[:, :-1, :]), 1)
            outputs += [future_pose, future_vel]

            if self.use_mask:
                assert 'future_mask' in seq, 'use_mask is true but there is no future_mask in dataset.'
                future_mask = torch.tensor(
                    [[seq.future_mask[person_idx][frame_idx]
                      for frame_idx in range(0, self.future_frames_num, self.skip_frame + 1)]
                     for person_idx in range(persons_num)])
                outputs.append(future_mask)

        return tuple(outputs)

    def get_tensor(self):
        future_mask = torch.tensor(
            [[seq.future_mask[person_idx][frame_idx]
              for frame_idx in range(0, self.future_frames_num, self.skip_frame + 1)]
             for person_idx in range(persons_num)])
