import torch

from utils.others import get_binary


def accuracy(pred, target, device):
    pred = get_binary(pred, device)

    return torch.sum(pred == target) / torch.numel(pred)


def f1_score(pred, target, device):
    pred = get_binary(pred, device)
    target_true = torch.sum(target)
    pred_true = torch.sum(pred)
    correct_true = torch.sum((target == 1) * (pred == 1))
    recall = correct_true / target_true
    precision = correct_true / pred_true
    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


def precision(pred, target, device):
    pred = get_binary(pred, device)
    pred_true = torch.sum(pred)
    correct_true = torch.sum((target == 1) * (pred == 1))

    return correct_true / pred_true


def recall(pred, target, device):
    pred = get_binary(pred, device)
    target_true = torch.sum(target)
    correct_true = torch.sum((target == 1) * (pred == 1))

    return correct_true / target_true
