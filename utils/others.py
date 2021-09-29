import torch
import numpy as np


def pose_from_vel(velocity, last_obs_pose, stay_in_frame=False):
    pose = torch.zeros_like(velocity).to('cuda')
    last_obs_pose_ = last_obs_pose

    for i in range(velocity.shape[-2]):
        pose[..., i, :] = last_obs_pose_ + velocity[..., i, :]
        last_obs_pose_ = pose[..., i, :]

    if stay_in_frame:
        for i in range(velocity.shape[-1]):
            pose[..., i] = torch.min(pose[..., i], 1920 * torch.ones_like(pose.shape[:-1]).to('cuda'))
            pose[..., i] = torch.max(pose[..., i], torch.zeros_like(pose.shape[:-1]).to('cuda'))

    return pose


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def get_binary(src, device):
    zero = torch.zeros_like(src).to(device)
    one = torch.ones_like(src).to(device)
    return torch.where(src > 0.5, one, zero)
