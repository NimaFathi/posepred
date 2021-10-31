import torch


def normalize(in_tensor, mean, std):
    bs, frame_n, feature_n = in_tensor.shape
    keypoint_dim = mean.shape[0]
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    assert mean.shape == std.shape
    assert feature_n % keypoint_dim == 0
    mean = mean.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)
    std = std.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)

    return (in_tensor - mean) / std


def denormalize(in_tensor, mean, std):
    bs, frame_n, feature_n = in_tensor.shape
    keypoint_dim = mean.shape[0]
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    assert mean.shape == std.shape
    assert feature_n % keypoint_dim == 0
    mean = mean.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)
    std = std.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)

    return (in_tensor * std) + mean
