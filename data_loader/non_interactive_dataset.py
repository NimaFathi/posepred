import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval


class NonInteractiveDataset(Dataset):
    def __init__(self, dataset_path, keypoint_dim, is_testing, use_mask, skip_frame):
        data = pd.read_csv(dataset_path)
        for col in list(data.columns[1:].values):
            try:
                data.loc[:, col] = data.loc[:, col].apply(lambda x: literal_eval(x))
            except:
                raise Exception("data must be convertable to valid data-scructures")

        self.data = data.copy().reset_index(drop=True)
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.skip_frame = skip_frame

        seq = self.data.iloc[0]
        self.keypoint_dim = keypoint_dim
        self.keypoints_num = int(len(seq.observed_pose[0]) / self.keypoint_dim)
        self.obs_frames_num = len(seq.observed_pose)
        if not self.is_testing:
            self.future_frames_num = len(seq.future_pose)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data.iloc[index]

        obs_pose = self.get_tensor(seq, 'observed_pose')
        obs_vel = (obs_pose[1:, :] - obs_pose[:-1, :])
        outputs = [obs_pose, obs_vel]

        if self.use_mask:
            obs_mask = self.get_tensor(seq, 'observed_mask')
            outputs.append(obs_mask)

        if not self.is_testing:
            assert len(seq.observed_pose) == len(seq.future_pose), "unequal persons in observed and future frames."
            future_pose = self.get_tensor(seq, 'future_pose')
            future_vel = torch.cat(
                ((future_pose[0, :] - obs_pose[-1, :]).unsqueeze(1), future_pose[1:, :] - future_pose[:-1, :]), 1)
            outputs += [future_pose, future_vel]

            if self.use_mask:
                future_mask = self.get_tensor(seq, 'future_mask')
                outputs.append(future_mask)

        return tuple(outputs)

    def get_tensor(self, seq, segment):
        assert segment in seq, 'No segment named: ' + segment
        frames_num = len(seq[segment])
        return torch.tensor([seq[segment][frame_idx] for frame_idx in range(0, frames_num, self.skip_frame + 1)])
