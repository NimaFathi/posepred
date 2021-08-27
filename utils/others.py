import torch


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
