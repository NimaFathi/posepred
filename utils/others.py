import torch


def pose_from_vel(pred_vel, last_obs_pose, stay_in_frame=False):
    pred_pose = torch.zeros_like(pred_vel).to('cuda')
    last_obs_pose_ = last_obs_pose

    for i in range(pred_vel.shape[-2]):
        pred_pose[..., i, :] = last_obs_pose_ + pred_vel[..., i, :]
        last_obs_pose_ = pred_pose[..., i, :]

    if stay_in_frame:
        for i in range(pred_vel.shape[-1]):
            pred_pose[..., i] = torch.min(pred_pose[..., i], 1920 * torch.ones_like(pred_pose.shape[:-1]).to('cuda'))
            pred_pose[..., i] = torch.max(pred_pose[..., i], torch.zeros_like(pred_pose.shape[:-1]).to('cuda'))

    return pred_pose
