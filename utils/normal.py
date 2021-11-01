import torch


def normalize(in_tensor, mean, std):
    device = 'cuda' if in_tensor.is_cuda else 'cpu'
    bs, frame_n, feature_n = in_tensor.shape
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    keypoint_dim = mean.shape[0]
    assert mean.shape == std.shape
    assert feature_n % keypoint_dim == 0
    mean = mean.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)
    std = std.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)

    return (in_tensor - mean) / std


def denormalize(in_tensor, mean, std):
    device = 'cuda' if in_tensor.is_cuda else 'cpu'
    bs, frame_n, feature_n = in_tensor.shape
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    keypoint_dim = mean.shape[0]
    assert mean.shape == std.shape
    assert feature_n % keypoint_dim == 0
    mean = mean.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)
    std = std.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)

    return (in_tensor * std) + mean
